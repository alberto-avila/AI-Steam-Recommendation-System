"""GenRec-style ranker: verbalized history -> Gemma 4 (QLoRA) -> pooled vector -> catalog scores.

Run ``python -m src.model --config config/model.yaml --smoke`` for the checkpoint 4
forward/backward pass. Mirrors the paper's scoring head: one prefill pass per user,
no token generation, and a dot product against a learned item-ID embedding table.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.verbalize import GemmaTokenizer, load_config as load_verbalize_config, prepare_tokenizer

LOGGER = logging.getLogger(__name__)
# Load the language model only; embedding tables are excluded here and sliced in by hand.
TEXT_PREFIX = r"^model\.language_model\.(?!embed_tokens)"
POOLING_MODES = ("mean", "last")


@dataclass(frozen=True)
class ModelConfig:
    seed: int
    verbalize_configs: tuple[Path, ...]
    smoke_prompts: Path
    artifacts_dir: Path
    results_dir: Path
    repository: str
    revision: str
    pooling: str
    item_dim: int
    lora: dict
    smoke: dict


def load_config(path: Path) -> ModelConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent
    if document["pooling"] not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}.")
    return ModelConfig(
        seed=int(document["seed"]),
        verbalize_configs=tuple((root / value).resolve() for value in document["verbalize_configs"]),
        smoke_prompts=(root / document["smoke_prompts"]).resolve(),
        artifacts_dir=(root / document["artifacts_dir"]).resolve(),
        results_dir=(root / document["results_dir"]).resolve(),
        repository=document["backbone"]["repository"],
        revision=document["backbone"]["revision"],
        pooling=document["pooling"],
        item_dim=int(document["item_dim"]),
        lora=dict(document["lora"]),
        smoke=dict(document["smoke"]),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    LOGGER.info("Seed: %s", seed)


# ---------------------------------------------------------------------------
# Vocabulary pruning
# ---------------------------------------------------------------------------

class PrunedVocab:
    """Map original Gemma token IDs onto a compact table of only the tokens we use.

    Why: Gemma 4 E2B carries ~2.3B parameters in embedding tables (262k vocab x 35
    per-layer embeddings). 4-bit quantization does not compress embeddings, so they
    would need ~5.5 GB on an 8 GB GPU. We never generate text (the LM head is unused),
    so rows for tokens that never appear in our prompts are dead weight. Embedding rows
    are frozen pretrained values, so pruning changes memory, not the model's function.
    """

    def __init__(self, kept_ids: Iterable[int], original_size: int):
        self.kept_ids = sorted(set(kept_ids))
        self.lookup = torch.full((original_size,), -1, dtype=torch.long)
        self.lookup[self.kept_ids] = torch.arange(len(self.kept_ids))

    def __len__(self) -> int:
        return len(self.kept_ids)

    def remap(self, token_ids: Sequence[int]) -> list[int]:
        mapped = self.lookup[torch.tensor(token_ids, dtype=torch.long)]
        if (mapped < 0).any():
            # Fail loudly: a silently mangled token would corrupt the prompt.
            missing = sorted(set(torch.tensor(token_ids)[mapped < 0].tolist()))
            raise KeyError(f"Tokens outside the pruned vocabulary: {missing[:10]}")
        return mapped.tolist()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def vocabulary_texts(config: ModelConfig) -> list[str]:
    """Every prompt we will score, plus each title in list context and all digits.

    Title templates cover prompts built later (e.g. training examples with a different
    subset of games); `PrunedVocab.remap` still raises if anything slips through.
    """
    texts = ["0123456789 0 1 2 3 4 5 6 7 8 9 % hrs games game"]
    for path in config.verbalize_configs:
        verbalize = load_verbalize_config(path)
        for split in ("validation", "test"):
            texts += [row["prompt"] for row in read_jsonl(verbalize["artifacts_dir"] / f"{split}_histories.jsonl")]
        with (verbalize["data"].processed_dir / "games.csv").open(encoding="utf-8", newline="") as handle:
            titles = [row["display_title"] for row in csv.DictReader(handle)]
        texts += [f"Most-played games:\n{title} (1 hrs), {title} (0.1 hrs)." for title in titles]
    return texts


def build_vocab(texts: Iterable[str], tokenizer: GemmaTokenizer, special_ids: Iterable[int]) -> PrunedVocab:
    kept = set(special_ids)
    for text in texts:
        kept.update(tokenizer.encode(text))
    return PrunedVocab(kept, tokenizer.backend.get_vocab_size())


def load_tokenizer(config: ModelConfig) -> tuple[GemmaTokenizer, int]:
    verbalize = load_verbalize_config(config.verbalize_configs[0])
    directory = prepare_tokenizer(verbalize)
    tokenizer = GemmaTokenizer(directory)
    settings = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    pad_id = tokenizer.backend.token_to_id(settings["pad_token"])
    return tokenizer, pad_id


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def load_backbone(config: ModelConfig, vocab: PrunedVocab, pad_id: int) -> nn.Module:
    """4-bit text-only Gemma 4 with a pruned vocabulary and LoRA adapters."""
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from safetensors import safe_open
    from transformers import AutoConfig, BitsAndBytesConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    text_config = AutoConfig.from_pretrained(config.repository, revision=config.revision).text_config
    # Shrinking the configured vocab makes the loader allocate small embedding tables;
    # the real rows are copied in below, so the full tables never reach the GPU.
    text_config.vocab_size = text_config.vocab_size_per_layer_input = len(vocab)
    text_config.pad_token_id = int(vocab.lookup[pad_id])
    backbone = Gemma4TextModel.from_pretrained(
        config.repository, revision=config.revision, config=text_config,
        # Gemma 4 is multimodal; load only the language model and skip vision/audio towers.
        # Unmapped keys (towers, full-size embeddings) are reported as unexpected and dropped.
        key_mapping={TEXT_PREFIX: ""},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    weights = hf_hub_download(config.repository, "model.safetensors", revision=config.revision)
    index = torch.tensor(vocab.kept_ids)
    with safe_open(weights, framework="pt") as handle, torch.no_grad():
        for name in ("embed_tokens", "embed_tokens_per_layer"):
            rows = handle.get_slice(f"model.language_model.{name}.weight")[:][index]
            getattr(backbone, name).weight.copy_(rows.to(getattr(backbone, name).weight))

    backbone = prepare_model_for_kbit_training(
        backbone, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = LoraConfig(r=config.lora["r"], lora_alpha=config.lora["alpha"],
                      lora_dropout=config.lora["dropout"],
                      target_modules=list(config.lora["target_modules"]), bias="none")
    return get_peft_model(backbone, lora)


# ---------------------------------------------------------------------------
# Scoring head
# ---------------------------------------------------------------------------

def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Collapse [B, T, d] hidden states into one user vector per row.

    mean: average over real tokens (padding excluded).
    last: the final real token, which in a causal model has attended to the whole prompt.
    Assumes right padding.
    """
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    if mode == "mean":
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    if mode == "last":
        last = attention_mask.sum(1).long() - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]
    raise ValueError(f"Unknown pooling mode: {mode}")


class GenRecModel(nn.Module):
    """Two towers: LLM user tower and an ID-embedding item tower, joined by a dot product.

    The item tower is a plain nn.Embedding over persistent game IDs, trained from
    scratch (as in the paper). That is what lets one forward pass score the whole
    catalog: item vectors do not depend on the user, so scoring is one matmul.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, n_items: int,
                 item_dim: int = 128, pooling: str = "mean"):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        self.backbone, self.pooling = backbone, pooling
        # LLM hidden size (1536) != item dim; a single linear map aligns the spaces.
        self.projection = nn.Linear(hidden_size, item_dim)
        self.item_embedding = nn.Embedding(n_items, item_dim)
        nn.init.normal_(self.item_embedding.weight, std=item_dim ** -0.5)

    def user_vector(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).last_hidden_state
        # Head runs in fp32: it is tiny, and bf16 dot products over 3.5k items lose precision.
        return self.projection(pool(hidden, attention_mask, self.pooling).float())

    def score(self, user: torch.Tensor, candidate_ids: torch.Tensor | None = None) -> torch.Tensor:
        """[B, n_items] catalog scores, or [B, C] scores for per-row candidates."""
        if candidate_ids is None:
            return user @ self.item_embedding.weight.T
        return torch.einsum("bd,bcd->bc", user, self.item_embedding(candidate_ids))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                candidate_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.score(self.user_vector(input_ids, attention_mask), candidate_ids)


def collate(token_lists: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the batch maximum; never truncate (prompt lengths are audited upstream)."""
    width = max(len(tokens) for tokens in token_lists)
    input_ids = torch.full((len(token_lists), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(token_lists), width), dtype=torch.long)
    for row, tokens in enumerate(token_lists):
        input_ids[row, :len(tokens)] = torch.tensor(tokens)
        attention_mask[row, :len(tokens)] = 1
    return input_ids, attention_mask


def parameter_counts(model: nn.Module) -> dict[str, int]:
    groups = {"lora": 0, "projection": 0, "item_embedding": 0, "other_trainable": 0, "frozen": 0}
    for name, parameter in model.named_parameters():
        # bitsandbytes packs two 4-bit weights per uint8 element.
        count = parameter.numel() * (2 if parameter.dtype == torch.uint8 else 1)
        if not parameter.requires_grad:
            groups["frozen"] += count
        elif "lora_" in name:
            groups["lora"] += count
        elif name.startswith(("projection", "item_embedding")):
            groups[name.split(".")[0]] += count
        else:
            groups["other_trainable"] += count
    return groups


# ---------------------------------------------------------------------------
# Checkpoint 4 smoke test
# ---------------------------------------------------------------------------

def build_model(config: ModelConfig, extra_texts: Iterable[str] = (),
                kept_ids: Sequence[int] | None = None) -> tuple[GenRecModel, GemmaTokenizer, PrunedVocab, int]:
    """Build the full ranker. `extra_texts` (e.g. training prompts) extend the vocabulary;
    `kept_ids` restores a saved vocabulary exactly when reloading a checkpoint."""
    tokenizer, pad_id = load_tokenizer(config)
    if kept_ids is not None:
        vocab = PrunedVocab(kept_ids, tokenizer.backend.get_vocab_size())
    else:
        texts = [*vocabulary_texts(config), *extra_texts]
        vocab = build_vocab(texts, tokenizer, special_ids=[pad_id, tokenizer.bos_id])
    LOGGER.info("Pruned vocabulary: %s of %s tokens", len(vocab), tokenizer.backend.get_vocab_size())
    verbalize = load_verbalize_config(config.verbalize_configs[0])
    mapping = json.loads((verbalize["data"].processed_dir / "game_mapping.json").read_text(encoding="utf-8"))
    backbone = load_backbone(config, vocab, pad_id)
    model = GenRecModel(backbone, backbone.config.hidden_size, n_items=len(mapping),
                        item_dim=config.item_dim, pooling=config.pooling)
    model.projection.cuda()
    model.item_embedding.cuda()
    return model, tokenizer, vocab, int(vocab.lookup[pad_id])


def smoke(config: ModelConfig) -> dict:
    seed_everything(config.seed)
    start = time.time()
    model, tokenizer, vocab, pad_index = build_model(config)
    load_seconds = time.time() - start
    counts = parameter_counts(model)
    LOGGER.info("Parameters: %s", counts)

    rows = read_jsonl(config.smoke_prompts)
    # Longest prompts first: memory is measured at the worst observed sequence length.
    rows.sort(key=lambda row: (-row["input_tokens"], row["user_id"]))
    n_items = model.item_embedding.num_embeddings
    generator = torch.Generator().manual_seed(config.seed)
    report = {"seed": config.seed, "pooling": config.pooling, "item_dim": config.item_dim,
              "hidden_size": model.projection.in_features, "n_items": n_items,
              "pruned_vocab": len(vocab), "original_vocab": tokenizer.backend.get_vocab_size(),
              "parameters": counts, "load_seconds": round(load_seconds, 1), "batches": []}
    model.train()
    for batch_size in config.smoke["batch_sizes"]:
        batch = rows[:batch_size]
        input_ids, attention_mask = collate(
            [vocab.remap(tokenizer.encode(row["prompt"])) for row in batch], pad_index)
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
        # Dummy targets: column 0 is the "positive", the rest random negatives.
        candidates = torch.randint(0, n_items, (batch_size, 1 + config.smoke["n_negatives"]),
                                   generator=generator).cuda()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tick = time.time()
        try:
            # One backbone pass; both scoring modes reuse the same user vector.
            user = model.user_vector(input_ids, attention_mask)
            full, sampled = model.score(user), model.score(user, candidates)
            loss = F.cross_entropy(sampled, torch.zeros(batch_size, dtype=torch.long, device="cuda"))
            loss.backward()
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            LOGGER.warning("Batch size %s: out of memory", batch_size)
            report["batches"].append({"batch_size": batch_size, "oom": True})
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        grads = {group: sum(float(p.grad.norm()) for n, p in model.named_parameters()
                            if p.grad is not None and match(n))
                 for group, match in (("lora", lambda n: "lora_" in n),
                                      ("projection", lambda n: n.startswith("projection")),
                                      ("item_embedding", lambda n: n.startswith("item_embedding")))}
        frozen_with_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
        entry = {"batch_size": batch_size, "seq_len": int(input_ids.shape[1]), "oom": False,
                 "full_scores_shape": list(full.shape), "sampled_scores_shape": list(sampled.shape),
                 "loss": loss.item(), "loss_finite": bool(torch.isfinite(loss.detach())),
                 "grad_norms": grads, "frozen_params_with_grad": len(frozen_with_grad),
                 "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                 "seconds": round(time.time() - tick, 2)}
        LOGGER.info("%s", entry)
        report["batches"].append(entry)
    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/model.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Run the checkpoint 4 forward/backward test.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.smoke:
        parser.error("Only --smoke is implemented at this checkpoint.")
    smoke(load_config(args.config))


if __name__ == "__main__":
    main()
