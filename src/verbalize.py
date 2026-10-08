"""Deterministic, leakage-checked histories and real Gemma token-length audits.

Run ``python -m src.verbalize --config config/verbalize.yaml``. Only the explicit
``--download-tokenizer`` flag permits network access; model weights are never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import tempfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from src.data import (
    ARTIFACT_NAMES, Interaction, build_game_mapping, clean_display_title, describe,
    leave_one_out, load_config as load_data_config, normalize_title, write_json,
)

LOGGER = logging.getLogger(__name__)
HEADER = "This player's most-played games:\n"
TOKEN_CONTRACT = "one BOS + plain history; no EOS, chat template, padding, or token truncation"
OUTPUT_FILES = {"test_histories.jsonl", "validation_histories.jsonl", "stats.json", "samples.md"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_hours(rows: Sequence[Interaction], mapping: dict[str, int]) -> list[Interaction]:
    """Use persistent IDs to make equal-hour selections independent of input order."""
    return sorted(rows, key=lambda row: (-row.hours, mapping[normalize_title(row.game_title)]))


def plain_header(kept: Sequence[Interaction], selected: Sequence[Interaction]) -> str:
    return HEADER


def summary_header(kept: Sequence[Interaction], selected: Sequence[Interaction]) -> str:
    """Restore whole-history signal that top-K drops: breadth, volume, and loyalty.

    Statistics use only the post-exclusion history, so held-out games cannot leak
    through counts or totals. The concentration line distinguishes a 50-game player
    who lives in one title from one who spreads time evenly; a count alone cannot.
    """
    total = sum(row.hours for row in kept)
    percent = round(100 * max(row.hours for row in kept) / total)
    # Never claim 100% (or 0%) when other games exist; rounding would misstate it.
    if len(kept) > 1:
        percent = min(percent, 99)
    share = "under 1%" if percent < 1 else f"{percent}%"
    games = f"{len(kept)} game" + ("s" if len(kept) != 1 else "")
    listed = (f"Most-played games (top {len(selected)} of {len(kept)}):\n"
              if len(selected) < len(kept) else "Most-played games:\n")
    return (f"This player has played {games} for {format_hours(round(total, 1))} hrs in total.\n"
            f"Their most-played game accounts for {share} of their playtime.\n" + listed)


# Each strategy is (ordering, header). Later ablations register here without
# changing the split/tokenization code.
STRATEGIES = {
    "top_hours": (top_hours, plain_header),
    "top_hours_summary": (top_hours, summary_header),
}


@dataclass(frozen=True)
class RenderedHistory:
    text: str
    game_ids: tuple[int, ...]
    source_games: int
    excluded_games: int
    omitted_by_k: int


def format_hours(hours: float) -> str:
    # Preserve short sessions (0.1 hours) instead of rounding them to zero.
    value = format(Decimal(str(hours)), "f")
    return value.rstrip("0").rstrip(".") if "." in value else value


def render_history(
    history: Sequence[Interaction], mapping: dict[str, int], display_titles: dict[int, str],
    *, excluded_game_ids: Iterable[int], k: int = 15, strategy: str = "top_hours",
) -> RenderedHistory:
    """Exclude labels BEFORE top-K. Labels and user identifiers never enter the text.

    Callers must supply every held-out ID for this user. For supervised training,
    this must also include the current training target; this function does not split.
    """
    if type(k) is not int or k < 1 or strategy not in STRATEGIES:
        raise ValueError("Expected a positive integer k and a registered strategy.")
    if len({row.user_id for row in history}) > 1:
        raise ValueError("A prompt must contain only one user's history.")
    excluded, seen, kept = set(excluded_game_ids), set(), []
    for row in history:
        title = normalize_title(row.game_title)
        if title not in mapping:
            raise ValueError(f"Unknown title: {row.game_title}")
        game_id = mapping[title]
        if game_id in seen or not math.isfinite(row.hours) or row.hours <= 0:
            raise ValueError("History has duplicate games or nonpositive/nonfinite hours.")
        seen.add(game_id)
        if game_id not in display_titles or normalize_title(display_titles[game_id]) != title:
            raise ValueError("Display title does not match its persistent game ID.")
        if game_id not in excluded:
            kept.append(row)
    order, header = STRATEGIES[strategy]
    selected = order(kept, mapping)[:k]
    if not selected:
        raise ValueError("Cannot render an empty history after exclusions.")
    ids = tuple(mapping[normalize_title(row.game_title)] for row in selected)
    entries = [f"{clean_display_title(display_titles[game_id])} ({format_hours(row.hours)} hrs)"
               for game_id, row in zip(ids, selected)]
    return RenderedHistory(header(kept, selected) + ", ".join(entries) + ".", ids, len(history),
                           len(history) - len(kept), len(kept) - len(selected))


def load_config(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"pipeline_config", "artifacts_dir", "results_dir", "strategy", "k",
                "max_tokens", "review_samples", "tokenizer"}
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError(f"Verbalizer config requires exactly {sorted(expected)}.")
    for key in ("k", "max_tokens", "review_samples"):
        if type(document[key]) is not int or document[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if document["strategy"] not in STRATEGIES:
        raise ValueError("Unknown verbalization strategy.")
    tokenizer = document["tokenizer"]
    if not isinstance(tokenizer, dict) or set(tokenizer) != {"repository", "revision", "directory", "sha256"}:
        raise ValueError("Tokenizer config requires repository, revision, directory, and sha256.")
    if not re.fullmatch(r"[\w-]+/[\w.-]+", str(tokenizer["repository"])) or not re.fullmatch(r"[a-f0-9]{40}", str(tokenizer["revision"])):
        raise ValueError("Pin a Hugging Face repository and full commit SHA.")
    hashes = tokenizer["sha256"]
    if not isinstance(hashes, dict) or set(hashes) != {"tokenizer.json", "tokenizer_config.json"} or any(
        not re.fullmatch(r"[a-f0-9]{64}", str(value)) for value in hashes.values()
    ):
        raise ValueError("Pin SHA-256 hashes for both tokenizer files.")
    root = path.resolve().parent.parent
    paths = {}
    for key, value in [(key, document[key]) for key in ("pipeline_config", "artifacts_dir", "results_dir")] + [("tokenizer_dir", tokenizer["directory"])]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path.")
        paths[key] = (root / value).resolve()
    data = load_data_config(paths["pipeline_config"])
    pipeline = yaml.safe_load(paths["pipeline_config"].read_text(encoding="utf-8"))
    baseline_root = paths["pipeline_config"].parent.parent
    baseline_dir = (baseline_root / pipeline.get("baselines", {}).get("artifacts_dir", "data/baselines")).resolve()
    inputs = [data.processed_dir.resolve(), data.raw_path.resolve().parent, baseline_dir]
    outputs = [paths[key] for key in ("artifacts_dir", "results_dir", "tokenizer_dir")]
    for index, output in enumerate(outputs):
        for other in inputs + outputs[index + 1:]:
            if output.is_relative_to(other) or other.is_relative_to(output):
                raise ValueError("Verbalizer outputs must not overlap inputs or each other.")
    return {"document": document, "data": data, "baseline_dir": baseline_dir, **paths}


def prepare_tokenizer(config: dict, download: bool = False) -> Path:
    directory = config["tokenizer_dir"]
    settings = config["document"]["tokenizer"]
    for name, expected in settings["sha256"].items():
        path = directory / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}. Run once with --download-tokenizer.")
            url = f"https://huggingface.co/{settings['repository']}/resolve/{settings['revision']}/{name}"
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Downloaded tokenizer checksum mismatch: {name}")
            directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(path)
        if file_hash(path) != expected:
            raise ValueError(f"Tokenizer checksum mismatch: {name}")
    return directory


class GemmaTokenizer:
    """An explicit reusable encoding contract for the future hidden-state ranker."""

    def __init__(self, directory: Path):
        from tokenizers import Tokenizer

        self.backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.backend.no_padding()
        self.backend.no_truncation()
        config = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.bos_id = self.backend.token_to_id(config["bos_token"])
        if self.bos_id is None:
            raise ValueError("Tokenizer is missing the configured BOS token.")

    def encode(self, text: str) -> list[int]:
        # The saved tokenizer has no automatic BOS. Add exactly one explicitly;
        # chat/generation scaffolding is unnecessary for our hidden-state scorer.
        return [self.bos_id] + self.backend.encode(text, add_special_tokens=False).ids

    def measure(self, text: str) -> dict[str, int]:
        length = len(self.encode(text))
        return {"text_tokens": length - 1, "input_tokens": length}


def read_rows(path: Path, mapping: dict[str, int]) -> list[Interaction]:
    """Read the shared CSV contract without importing optional ALS dependencies."""
    rows, seen = [], set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "game_id", "game_title", "hours"]:
            raise ValueError(f"Unexpected interaction schema: {path.name}")
        for raw in reader:
            row = Interaction(raw["user_id"], raw["game_title"], float(raw["hours"]))
            pair = (row.user_id, row.game_title)
            if not row.user_id or row.game_title not in mapping or int(raw["game_id"]) != mapping[row.game_title]:
                raise ValueError(f"Invalid user or game ID/title in {path.name}")
            if pair in seen or not math.isfinite(row.hours) or row.hours <= 0:
                raise ValueError(f"Duplicate pair or invalid hours in {path.name}")
            seen.add(pair)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty interactions: {path}")
    return rows


def load_sources(config: dict) -> dict:
    directory, baseline = config["data"].processed_dir, config["baseline_dir"]
    input_hashes = {name: file_hash(directory / name) for name in ARTIFACT_NAMES}
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    if input_hashes != manifest["identity"]["input_sha256"]:
        raise ValueError("Processed data differs from the completed baseline checkpoint.")
    baseline_hashes = {name: file_hash(baseline / name) for name in (
        "validation_fit.csv", "validation.csv", "user_mapping.json",
    )}
    if any(digest != manifest["artifact_sha256"][name] for name, digest in baseline_hashes.items()):
        raise ValueError("Baseline validation membership or user mapping was modified.")
    mapping = build_game_mapping([], directory / "game_mapping.json")
    train, test = [read_rows(directory / name, mapping) for name in ("train.csv", "test.csv")]
    fit, validation = [read_rows(baseline / name, mapping) for name in ("validation_fit.csv", "validation.csv")]
    if set(train) & set(test) or set(fit) & set(validation) or set(fit + validation) != set(train):
        raise ValueError("Split overlap or changed training membership/hours.")
    # Check game identity, not substrings: 'Portal' and 'Portal 2' are different games.
    pairs = lambda rows: {(row.user_id, row.game_title) for row in rows}
    if pairs(train) & pairs(test) or pairs(fit) & pairs(validation):
        raise ValueError("Held-out game leaked into fitting rows.")
    expected_fit, expected_validation = leave_one_out(train, config["data"].seed + 1)
    if set(fit) != set(expected_fit) or set(validation) != set(expected_validation):
        raise ValueError("Validation differs from the baseline's deterministic split.")
    users = sorted({row.user_id for row in train})
    for targets in (test, validation):
        if Counter(row.user_id for row in targets) != dict.fromkeys(users, 1):
            raise ValueError("Expected exactly one target per user.")
    if json.loads((baseline / "user_mapping.json").read_text()) != dict(zip(users, range(len(users)))):
        raise ValueError("Baseline user indexing differs.")
    stats = json.loads((directory / "stats.json").read_text())
    if stats["seed"] != config["data"].seed or stats["min_games_per_user"] != config["data"].min_games_per_user:
        raise ValueError("Pipeline settings differ from the existing data checkpoint.")
    with (directory / "games.csv").open(encoding="utf-8", newline="") as handle:
        display = {int(row["game_id"]): row["display_title"] for row in csv.DictReader(handle)}
    return {"train": train, "test": test, "fit": fit, "validation": validation,
            "mapping": mapping, "display": display, "users": users,
            "input_hashes": input_hashes, "baseline_hashes": baseline_hashes}


def build_records(sources: dict, tokenizer: GemmaTokenizer, *, split: str, k: int,
                  strategy: str, max_tokens: int) -> list[dict]:
    if split not in ("test", "validation"):
        raise ValueError("Expected test or validation split.")
    histories, excluded = defaultdict(list), defaultdict(set)
    for row in sources["train" if split == "test" else "fit"]:
        histories[row.user_id].append(row)
    for row in sources["test"] + (sources["validation"] if split == "validation" else []):
        excluded[row.user_id].add(sources["mapping"][row.game_title])
    targets = {row.user_id: sources["mapping"][row.game_title] for row in sources[split]}
    records = []
    for user in sources["users"]:
        rendered = render_history(histories[user], sources["mapping"], sources["display"],
                                  excluded_game_ids=excluded[user], k=k, strategy=strategy)
        tokens = tokenizer.measure(rendered.text)
        records.append({"user_id": user, "split": split, "target_game_id": targets[user],
                        "excluded_game_ids": sorted(excluded[user]), "prompt": rendered.text,
                        "game_ids": list(rendered.game_ids), "history_games": rendered.source_games,
                        "excluded_from_source": rendered.excluded_games,
                        "rendered_games": len(rendered.game_ids), "omitted_by_k": rendered.omitted_by_k,
                        **tokens, "over_budget": tokens["input_tokens"] > max_tokens})
    return records


def select_samples(records: Sequence[dict], count: int, seed: int) -> list[dict]:
    """Cover short through long histories; seed-derived ties avoid cherry-picking."""
    ordered = sorted(records, key=lambda row: (
        row["history_games"], hashlib.sha256(f"{seed}:{row['user_id']}".encode()).hexdigest(),
    ))
    count = min(count, len(ordered))
    if count < 1:
        return []
    return [ordered[round(index * (len(ordered) - 1) / max(1, count - 1))] for index in range(count)]


def summarize_records(records: Sequence[dict]) -> dict:
    return {"users": len(records),
            **{key: describe(row[key] for row in records) for key in (
                "history_games", "rendered_games", "omitted_by_k", "text_tokens", "input_tokens")},
            "histories_capped_by_k": sum(row["omitted_by_k"] > 0 for row in records),
            "games_omitted_by_k": sum(row["omitted_by_k"] for row in records),
            "over_budget": sum(row["over_budget"] for row in records),
            "excluded_from_source": sum(row["excluded_from_source"] for row in records)}


def samples_markdown(samples: Sequence[dict], display: dict[int, str], stats: dict) -> str:
    lines = [f"# Verbalizer checkpoint: {len(samples)} histories for review", "",
             f"Strategy: {stats['strategy']}; up to {stats['k']} games. Source: original training rows only.",
             "Samples span training-history sizes, with seed-derived tie ordering. They were not chosen by model scores.",
             "Target names below are audit labels outside the prompt. No scores or target hours enter the prompt.",
             "", f"Token contract: {TOKEN_CONTRACT}.",
             f"Token budget: {stats['max_tokens']}. Prompts exceeding it are flagged, never silently truncated.", "",
             "| Split | Users | Median tokens | p99 tokens | Max tokens | Over budget | Histories capped by K |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, values in stats["splits"].items():
        lengths = values["input_tokens"]
        lines.append(f"| {split} | {values['users']} | {lengths['median']:g} | {lengths['p99']:g} | {lengths['max']} | {values['over_budget']} | {values['histories_capped_by_k']} |")
    for number, row in enumerate(samples, 1):
        lines += ["", f"## {number}. User {row['user_id']}", "",
                  f"Training games: {row['history_games']}; rendered: {row['rendered_games']}; omitted by K: {row['omitted_by_k']}; input tokens: {row['input_tokens']}.",
                  f"Held-out test label (excluded): **{display[row['target_game_id']]}** (ID {row['target_game_id']}).", "",
                  "```text", row["prompt"], "```"]
    return "\n".join(lines) + "\n"


def make_identity(config: dict, sources: dict) -> dict:
    return {"schema_version": 1, "configuration": config["document"], "seed": config["data"].seed,
            "input_sha256": sources["input_hashes"], "baseline_sha256": sources["baseline_hashes"],
            "code_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in ("verbalize.py", "data.py")},
            "versions": {"python": platform.python_version(), **{
                name: importlib.metadata.version(name) for name in ("tokenizers", "PyYAML")}},
            "token_contract": TOKEN_CONTRACT,
            "sample_selection": "evenly spaced positions by history length; SHA256(seed:user_id) ties"}


def verify_artifacts(directory: Path, expected_identity: dict | None = None) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise ValueError("Incompatible verbalizer cache; choose a new artifacts_dir for changed settings/code/inputs.")
    if set(manifest["artifact_sha256"]) != OUTPUT_FILES:
        raise ValueError("Incomplete verbalizer manifest.")
    for name, expected in manifest["artifact_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Modified verbalizer artifact: {name}")
    return manifest


def run_verbalizer(config: dict, download_tokenizer: bool = False) -> dict:
    sources = load_sources(config)
    directory = prepare_tokenizer(config, download=download_tokenizer)
    identity = make_identity(config, sources)
    output, settings = config["artifacts_dir"], config["document"]
    if output.exists():
        verify_artifacts(output, identity)
        LOGGER.info("Reusing verified verbalizer artifacts: %s", output)
    else:
        tokenizer = GemmaTokenizer(directory)
        records = {split: build_records(sources, tokenizer, split=split, **{
            name: settings[name] for name in ("k", "strategy", "max_tokens")
        }) for split in ("validation", "test")}
        samples = select_samples(records["test"], settings["review_samples"], config["data"].seed)
        stats = {"seed": config["data"].seed, "strategy": settings["strategy"], "k": settings["k"],
                 "max_tokens": settings["max_tokens"], "token_contract": TOKEN_CONTRACT,
                 "tokenizer": settings["tokenizer"], "sample_user_ids": [row["user_id"] for row in samples],
                 "splits": {split: summarize_records(rows) for split, rows in records.items()}}
        # Publish a complete bundle; interrupted runs never look like a finished cache.
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="verbalizer-", dir=output.parent) as temporary:
            staged = Path(temporary) / "artifacts"
            staged.mkdir()
            for split, rows in records.items():
                payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
                (staged / f"{split}_histories.jsonl").write_text(payload, encoding="utf-8", newline="\n")
            write_json(staged / "stats.json", stats)
            (staged / "samples.md").write_text(samples_markdown(samples, sources["display"], stats), encoding="utf-8", newline="\n")
            write_json(staged / "manifest.json", {"identity": identity, "artifact_sha256": {
                name: file_hash(staged / name) for name in sorted(OUTPUT_FILES)
            }})
            verify_artifacts(staged, identity)
            staged.rename(output)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    for name in ("samples.md", "stats.json"):
        (config["results_dir"] / name).write_bytes((output / name).read_bytes())
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    for split, values in stats["splits"].items():
        LOGGER.info("%s: %s users; tokens %s; over %s tokens: %s; histories capped by K: %s",
                    split, values["users"], values["input_tokens"], settings["max_tokens"],
                    values["over_budget"], values["histories_capped_by_k"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/verbalize.yaml"))
    parser.add_argument("--download-tokenizer", action="store_true", help="Download only missing pinned tokenizer files.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_verbalizer(load_config(args.config), args.download_tokenizer)


if __name__ == "__main__":
    main()
