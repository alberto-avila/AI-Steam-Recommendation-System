"""QLoRA training loop for the GenRec ranker.

    python -m src.train --config config/train.yaml --overfit   # checkpoint 5: plumbing test
    python -m src.train --config config/train.yaml             # development run (fit -> validation)

Training examples are leave-one-out over each user's fitting history: every played game
becomes a target once, with that game removed from its own prompt. This matches how the
evaluation target was drawn (uniformly from the user's games).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.evaluate import EvaluationCase, evaluate_model, prepare_cases, summarize
from src.model import (
    GenRecModel, GemmaTokenizer, PrunedVocab, build_model, collate, load_config as load_model_config,
    parameter_counts, read_jsonl, seed_everything,
)
from src.verbalize import load_config as load_verbalize_config, load_sources, render_history

LOGGER = logging.getLogger(__name__)
LOSSES = ("sampled", "full")


@dataclass(frozen=True)
class Example:
    user_id: str
    target: int
    prompt: str


def load_config(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document["loss"] not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}.")
    root = path.resolve().parent.parent
    for key in ("model_config", "verbalize_config", "artifacts_dir", "results_dir"):
        document[key] = (root / document[key]).resolve()
    return document


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def training_examples(sources: dict, *, k: int, strategy: str) -> list[Example]:
    """One example per fitting interaction, target removed from its own prompt.

    Validation and test targets are also excluded, so no held-out game can appear in
    any training prompt or its summary statistics.
    """
    mapping, display = sources["mapping"], sources["display"]
    histories: dict[str, list] = {}
    for row in sources["fit"]:
        histories.setdefault(row.user_id, []).append(row)
    held_out: dict[str, set[int]] = {}
    for row in sources["validation"] + sources["test"]:
        held_out.setdefault(row.user_id, set()).add(mapping[row.game_title])
    examples = []
    for user in sources["users"]:
        for row in histories[user]:
            target = mapping[row.game_title]
            rendered = render_history(histories[user], mapping, display, k=k, strategy=strategy,
                                      excluded_game_ids=held_out[user] | {target})
            examples.append(Example(user, target, rendered.text))
    return examples


def known_positives(sources: dict) -> dict[str, set[int]]:
    """Every observed game per user: used only to keep positives out of negative pools."""
    known: dict[str, set[int]] = {}
    for row in sources["train"] + sources["test"]:
        known.setdefault(row.user_id, set()).add(sources["mapping"][row.game_title])
    return known


def popularity(rows: Sequence, mapping: dict[str, int]) -> np.ndarray:
    """Distinct users per game in the fitting rows (same definition as the baseline)."""
    return np.bincount([mapping[row.game_title] for row in rows], minlength=len(mapping)).astype(np.float64)


def validation_cases(sources: dict, counts: np.ndarray, known: dict[str, set[int]],
                     seed: int, baseline_dir: Path) -> list[EvaluationCase]:
    """Rebuild the baseline's validation candidates and prove they are identical."""
    targets = {row.user_id: sources["mapping"][row.game_title] for row in sources["validation"]}
    cases = prepare_cases(targets, counts, known, seed=seed, n_negatives=100,
                          exponent=0.75, namespace="validation")
    saved = read_jsonl(baseline_dir / "validation_candidates.jsonl")
    if [case.sampled_record() for case in cases] != saved:
        raise ValueError("Validation candidates differ from the ALS baseline's saved candidates.")
    return cases


class NegativeSampler:
    """popularity^0.75 negatives from games the user never played.

    Uniform negatives are mostly obscure games that are trivially easy to rank below the
    positive; popularity-weighted ones are harder and closer to real competitors.
    """

    def __init__(self, counts: np.ndarray, known: dict[str, set[int]], exponent: float, seed: int):
        self.weights = torch.tensor(counts, dtype=torch.float64) ** exponent
        self.known = {user: torch.tensor(sorted(games)) for user, games in known.items()}
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, user_id: str, n: int) -> torch.Tensor:
        weights = self.weights.clone()
        weights[self.known[user_id]] = 0
        return torch.multinomial(weights, n, replacement=False, generator=self.generator)


def full_catalog_mask(batch: Sequence[Example], known: dict[str, set[int]],
                      eligible: torch.Tensor) -> torch.Tensor:
    """True where an item must not compete: the user's other positives, or items with no
    fitting interactions (never candidates at evaluation either)."""
    mask = (~eligible).unsqueeze(0).repeat(len(batch), 1)
    for row, example in enumerate(batch):
        others = [game for game in known[example.user_id] if game != example.target]
        mask[row, others] = True
    return mask


# ---------------------------------------------------------------------------
# Evaluation adapter
# ---------------------------------------------------------------------------

class CatalogScorer:
    """Adapts the ranker to the shared `score(user_id, game_ids)` evaluator interface.

    One forward pass per user scores the entire catalog; lookups then index that row.
    """

    def __init__(self, scores: np.ndarray, users: Sequence[str]):
        self.scores, self.rows = scores, {user: index for index, user in enumerate(users)}

    def score(self, user_id: str, game_ids: Sequence[int]) -> np.ndarray:
        return self.scores[self.rows[user_id], np.asarray(game_ids)]


@torch.no_grad()
def catalog_scores(model: GenRecModel, prompts: Sequence[str], tokenizer: GemmaTokenizer,
                   vocab: PrunedVocab, pad_index: int, batch_size: int) -> np.ndarray:
    model.eval()
    encoded = [vocab.remap(tokenizer.encode(prompt)) for prompt in prompts]
    # Length-sorted batches waste less compute on padding; order is restored below.
    order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]))
    scores = np.empty((len(encoded), model.item_embedding.num_embeddings), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        input_ids, mask = collate([encoded[index] for index in indices], pad_index)
        scores[indices] = model(input_ids.cuda(), mask.cuda()).float().cpu().numpy()
    model.train()
    return scores


def validate(model, records, cases, tokenizer, vocab, pad_index, batch_size) -> dict:
    users = [row["user_id"] for row in records]
    scorer = CatalogScorer(catalog_scores(model, [row["prompt"] for row in records],
                                          tokenizer, vocab, pad_index, batch_size), users)
    summary = summarize(evaluate_model(scorer, cases, model_name="genrec", split="validation"))
    return {f"{row['protocol']}_{key}": row[key] for row in summary
            if row["population"] == "all_users" for key in ("recall_at_10", "mrr")}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config: dict, model_config, model: GenRecModel, tokenizer, vocab, pad_index,
                 sampler: NegativeSampler, known, eligible: torch.Tensor):
        self.config, self.model_config, self.model = config, model_config, model
        self.tokenizer, self.vocab, self.pad_index = tokenizer, vocab, pad_index
        self.sampler, self.known, self.eligible = sampler, known, eligible

    def make_optimizer(self, total_steps: int):
        from transformers import get_cosine_schedule_with_warmup

        lora = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora_" in n]
        head = [*self.model.projection.parameters(), *self.model.item_embedding.parameters()]
        # Separate rates: LoRA nudges a pretrained network; the head starts from random
        # init and must learn 3.5k item vectors in about one epoch.
        optimizer = torch.optim.AdamW([
            {"params": lora, "lr": self.config["lr"]},
            {"params": head, "lr": self.config["head_lr"]},
        ], weight_decay=self.config["weight_decay"])
        warmup = math.ceil(self.config["warmup_ratio"] * total_steps)
        return optimizer, get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    def loss(self, batch: Sequence[Example]) -> torch.Tensor:
        input_ids, mask = collate([self.vocab.remap(self.tokenizer.encode(ex.prompt)) for ex in batch],
                                  self.pad_index)
        user = self.model.user_vector(input_ids.cuda(), mask.cuda())
        targets = torch.tensor([ex.target for ex in batch])
        if self.config["loss"] == "sampled":
            negatives = torch.stack([self.sampler.sample(ex.user_id, self.config["n_negatives"]) for ex in batch])
            # Column 0 is the positive, so the cross-entropy label is always 0.
            candidates = torch.cat([targets.unsqueeze(1), negatives], dim=1).cuda()
            scores = self.model.score(user, candidates)
            return F.cross_entropy(scores, torch.zeros(len(batch), dtype=torch.long, device="cuda"))
        scores = self.model.score(user)
        scores = scores.masked_fill(full_catalog_mask(batch, self.known, self.eligible).cuda(), float("-inf"))
        return F.cross_entropy(scores, targets.cuda())

    def train(self, examples: list[Example], epochs: int, *, eval_fn=None, log_every: int = 20) -> dict:
        cfg = self.config
        batch_size, accumulation = cfg["batch_size"], cfg["grad_accum"]
        steps_per_epoch = math.ceil(len(examples) / (batch_size * accumulation))
        total_steps = steps_per_epoch * epochs
        optimizer, scheduler = self.make_optimizer(total_steps)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        rng = random.Random(self.model_config.seed)
        history, step, start = [], 0, time.time()
        self.model.train()
        for epoch in range(epochs):
            order = list(examples)
            rng.shuffle(order)
            micro = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
            for first in range(0, len(micro), accumulation):
                group = micro[first:first + accumulation]
                n_group = sum(len(batch) for batch in group)
                total = 0.0
                for batch in group:
                    # Weight by batch size so a short final batch is not over-counted.
                    loss = self.loss(batch) * (len(batch) / n_group)
                    loss.backward()
                    total += loss.item()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"]).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                entry = {"step": step, "epoch": epoch + (first + len(group)) / len(micro),
                         "loss": total, "grad_norm": grad_norm, "lr": scheduler.get_last_lr()[0],
                         "elapsed_s": round(time.time() - start, 1)}
                if eval_fn and (step % cfg["eval_every_steps"] == 0 or step == total_steps):
                    entry["validation"] = eval_fn()
                history.append(entry)
                if step % log_every == 0 or step == total_steps or "validation" in entry:
                    rate = step * batch_size * accumulation / (time.time() - start)
                    LOGGER.info("step %s/%s epoch %.2f loss %.4f grad %.2f (%.1f ex/s)%s", step, total_steps,
                                entry["epoch"], total, grad_norm, rate,
                                f" validation {entry['validation']}" if "validation" in entry else "")
        return {"history": history, "total_steps": total_steps,
                "seconds": round(time.time() - start, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


@torch.no_grad()
def training_set_ranks(trainer: Trainer, examples: Sequence[Example]) -> dict:
    """Full-catalog rank of each target among eligible games (other positives excluded)."""
    prompts = [ex.prompt for ex in examples]
    scores = torch.from_numpy(catalog_scores(trainer.model, prompts, trainer.tokenizer, trainer.vocab,
                                             trainer.pad_index, trainer.config["eval_batch_size"]))
    scores = scores.masked_fill(full_catalog_mask(examples, trainer.known, trainer.eligible), float("-inf"))
    targets = torch.tensor([ex.target for ex in examples])
    target_scores = scores.gather(1, targets.unsqueeze(1))
    ranks = (scores > target_scores).sum(1) + 1
    loss = F.cross_entropy(scores, targets).item()
    return {"full_catalog_loss": loss, "recall_at_10": (ranks <= 10).float().mean().item(),
            "mrr": (1.0 / ranks.float()).mean().item(), "median_rank": ranks.median().item()}


def setup(config: dict, overfit: bool):
    model_config = load_model_config(config["model_config"])
    seed_everything(model_config.seed)
    verbalize = load_verbalize_config(config["verbalize_config"])
    settings = verbalize["document"]
    sources = load_sources(verbalize)
    examples = training_examples(sources, k=settings["k"], strategy=settings["strategy"])
    LOGGER.info("Training examples: %s (strategy %s, K=%s)", len(examples), settings["strategy"], settings["k"])
    if overfit:
        examples = random.Random(model_config.seed).sample(examples, config["overfit"]["n_examples"])
    counts = popularity(sources["fit"], sources["mapping"])
    known = known_positives(sources)
    sampler = NegativeSampler(counts, known, config["popularity_exponent"], model_config.seed)
    eligible = torch.from_numpy(counts > 0)
    validation_records = read_jsonl(verbalize["artifacts_dir"] / "validation_histories.jsonl")
    model, tokenizer, vocab, pad_index = build_model(
        model_config, extra_texts=[ex.prompt for ex in examples] + [row["prompt"] for row in validation_records])
    trainer = Trainer(config, model_config, model, tokenizer, vocab, pad_index, sampler, known, eligible)
    return trainer, examples, sources, verbalize, counts, known, validation_records


def run_overfit(config: dict) -> dict:
    trainer, examples, *_ = setup(config, overfit=True)
    LOGGER.info("Parameters: %s", parameter_counts(trainer.model))
    before = training_set_ranks(trainer, examples)
    LOGGER.info("Before training (on the %s examples): %s", len(examples), before)
    result = trainer.train(examples, config["overfit"]["epochs"], log_every=10)
    after = training_set_ranks(trainer, examples)
    LOGGER.info("After training: %s", after)
    report = {"loss": config["loss"], "n_examples": len(examples), "epochs": config["overfit"]["epochs"],
              "chance_loss_sampled": math.log(1 + config["n_negatives"]),
              "before": before, "after": after, **result}
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    path = config["results_dir"] / f"overfit_{config['loss']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return report


def run_development(config: dict) -> dict:
    trainer, examples, sources, verbalize, counts, known, records = setup(config, overfit=False)
    cases = validation_cases(sources, counts, known, trainer.model_config.seed, verbalize["baseline_dir"])
    if [row["user_id"] for row in records] != [case.user_id for case in cases]:
        raise ValueError("Validation prompts and candidates are not aligned by user.")
    evaluate = lambda: validate(trainer.model, records, cases, trainer.tokenizer, trainer.vocab,
                                trainer.pad_index, config["eval_batch_size"])
    result = trainer.train(examples, config["epochs"], eval_fn=evaluate)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    (config["results_dir"] / f"development_{config['loss']}.json").write_text(
        json.dumps({"loss": config["loss"], **result}, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    parser.add_argument("--overfit", action="store_true", help="Checkpoint 5: memorize a tiny training set.")
    parser.add_argument("--loss", choices=LOSSES, help="Override the configured loss.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    if args.loss:
        config["loss"] = args.loss
    (run_overfit if args.overfit else run_development)(config)


if __name__ == "__main__":
    main()
