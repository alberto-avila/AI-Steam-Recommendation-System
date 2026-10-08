"""Prepare Steam-200k with stable item IDs and random leave-one-out evaluation.

Run from the repository root: ``python -m src.data --config config/default.yaml``.
This stage uses the standard library plus PyYAML; no training dependencies are needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

LOGGER = logging.getLogger(__name__)
SOURCE_URL = "https://www.kaggle.com/datasets/tamber/steam-video-games"
ARTIFACT_NAMES = (
    "interactions.csv", "train.csv", "test.csv", "games.csv",
    "game_mapping.json", "stats.json",
)


@dataclass(frozen=True)
class Interaction:
    user_id: str
    game_title: str
    hours: float


@dataclass(frozen=True)
class PipelineConfig:
    raw_path: Path
    processed_dir: Path
    min_games_per_user: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if type(self.min_games_per_user) is not int or self.min_games_per_user < 2:
            raise ValueError("min_games_per_user must be an integer >= 2.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if self.raw_path.resolve() in {
            (self.processed_dir / name).resolve() for name in ARTIFACT_NAMES
        }:
            raise ValueError("The raw input cannot also be a pipeline output file.")


def load_config(path: Path) -> PipelineConfig:
    """Resolve paths independently of the caller's working directory."""
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    required = {"seed", "data"}
    allowed = required | {"baselines", "evaluation"}
    if not isinstance(document, dict) or not required <= set(document) or set(document) - allowed:
        raise ValueError("Config requires 'seed' and 'data'; optional sections are 'baselines' and 'evaluation'.")
    for section in ("baselines", "evaluation"):
        if section in document and not isinstance(document[section], dict):
            raise ValueError(f"Config '{section}' must be a mapping.")
    data = document["data"]
    expected = {"raw_path", "processed_dir", "min_games_per_user"}
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"Config 'data' must contain exactly {sorted(expected)}.")
    for key in ("raw_path", "processed_dir"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"Config data.{key} must be a nonempty path string.")
    project_root = path.resolve().parent.parent
    return PipelineConfig(
        raw_path=project_root / data["raw_path"],
        processed_dir=project_root / data["processed_dir"],
        min_games_per_user=data["min_games_per_user"],
        seed=document["seed"],
    )


def clean_display_title(title: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", title).split())


def normalize_title(title: str) -> str:
    """Unify Unicode, whitespace, and casing without removing title punctuation."""
    return clean_display_title(title).casefold()


def load_interactions(path: Path) -> tuple[list[Interaction], dict, dict[str, str]]:
    """Read four/five-column CSV, keep positive play, and deduplicate before splitting."""
    counts = Counter({
        "n_rows": 0, "n_purchase_rows_dropped": 0, "n_play_rows": 0,
        "n_nonpositive_play_rows_dropped": 0, "n_duplicate_play_rows_merged": 0,
    })
    pairs: dict[tuple[str, str], float] = {}
    title_variants: dict[str, Counter] = defaultdict(Counter)
    first_row = True
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, strict=True):
            if not row or all(not value.strip() for value in row):
                continue
            # The public file is headerless; also accept its documented column names.
            if first_row and [value.strip().lower() for value in row[:4]] == [
                "user-id", "game-title", "behavior-name", "value",
            ]:
                first_row = False
                continue
            first_row = False
            counts["n_rows"] += 1
            row_number = counts["n_rows"]
            if len(row) not in (4, 5):
                raise ValueError(f"Data row {row_number}: expected 4 or 5 columns, got {len(row)}.")
            # Kaggle's fifth field is an unused placeholder, not an interaction feature.
            user_id, title, behavior, value = (field.strip() for field in row[:4])
            if not user_id.isascii() or not user_id.isdecimal() or int(user_id) <= 0:
                raise ValueError(f"Data row {row_number}: user ID must be a positive integer.")
            user_id = str(int(user_id))  # Keep IDs as strings to avoid float precision loss.
            normalized = normalize_title(title)
            if not normalized:
                raise ValueError(f"Data row {row_number}: game title is empty.")
            if behavior == "purchase":
                counts["n_purchase_rows_dropped"] += 1
                continue
            if behavior != "play":
                raise ValueError(f"Data row {row_number}: unknown behavior {behavior!r}.")
            counts["n_play_rows"] += 1
            try:
                hours = float(value)
            except ValueError as exc:
                raise ValueError(f"Data row {row_number}: invalid play hours {value!r}.") from exc
            if not math.isfinite(hours):
                raise ValueError(f"Data row {row_number}: play hours must be finite.")
            if hours <= 0:
                counts["n_nonpositive_play_rows_dropped"] += 1
                continue
            pair = (user_id, normalized)
            if pair in pairs:
                counts["n_duplicate_play_rows_merged"] += 1
            # Hours are cumulative snapshots: summing duplicates would inflate engagement.
            pairs[pair] = max(pairs.get(pair, 0.0), hours)
            title_variants[normalized][clean_display_title(title)] += 1
    interactions = [Interaction(user, title, hours) for (user, title), hours in sorted(pairs.items())]
    # Preserve a readable title for the later verbalizer; break frequency ties lexically.
    display_titles = {
        title: min(variants, key=lambda variant: (-variants[variant], variant))
        for title, variants in title_variants.items()
    }
    return interactions, dict(counts), display_titles


def filter_users(interactions: list[Interaction], minimum: int) -> list[Interaction]:
    """Input has one row per user/game, so counts here represent distinct played games."""
    counts = Counter(row.user_id for row in interactions)
    return [row for row in interactions if counts[row.user_id] >= minimum]


def build_game_mapping(titles: Iterable[str], path: Path) -> dict[str, int]:
    """Reuse IDs; append sorted new titles without renumbering or deleting existing IDs."""
    mapping: dict[str, int] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            mapping = json.load(handle)
        if not isinstance(mapping, dict) or any(
            not title or normalize_title(title) != title or type(game_id) is not int
            for title, game_id in mapping.items()
        ):
            raise ValueError("Existing game mapping must map normalized titles to integer IDs.")
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Existing game mapping IDs must be unique and contiguous from zero.")
    for title in sorted(set(titles) - mapping.keys()):
        mapping[title] = len(mapping)
    return mapping


def leave_one_out(
    interactions: list[Interaction], seed: int,
) -> tuple[list[Interaction], list[Interaction]]:
    """Uniformly hold out one distinct game per user; Steam-200k has no timestamps."""
    histories: dict[str, list[Interaction]] = defaultdict(list)
    seen = set()
    for row in interactions:
        pair = (row.user_id, row.game_title)
        if pair in seen:
            raise ValueError("Deduplicate user/game pairs before leave-one-out splitting.")
        seen.add(pair)
        histories[row.user_id].append(row)
    train, test = [], []
    for user_id in sorted(histories):
        history = sorted(histories[user_id], key=lambda row: row.game_title)
        if len(history) < 2:
            raise ValueError(f"User {user_id} needs at least two games for leave-one-out.")
        # A user-specific seed makes splits stable when unrelated users are added/removed.
        held_out = random.Random(f"{seed}:{user_id}").randrange(len(history))
        test.append(history[held_out])
        train.extend(row for index, row in enumerate(history) if index != held_out)
    return train, test


def describe(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered), "min": ordered[0], "max": ordered[-1],
        "mean": statistics.mean(ordered), "std": statistics.pstdev(ordered),
        "p25": percentile(0.25), "median": percentile(0.5),
        "p75": percentile(0.75), "p90": percentile(0.9), "p99": percentile(0.99),
    }


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def run_pipeline(config: PipelineConfig) -> dict:
    LOGGER.info("Preparing Steam-200k with seed=%d", config.seed)
    if not config.raw_path.is_file():
        raise FileNotFoundError(
            f"Steam-200k CSV not found: {config.raw_path}. "
            f"Download steam-200k.csv from {SOURCE_URL}; see README.md."
        )
    interactions, raw_stats, display_titles = load_interactions(config.raw_path)
    filtered = filter_users(interactions, config.min_games_per_user)
    if not filtered:
        raise ValueError("No interactions remain after filtering. Check the CSV and minimum games.")
    mapping = build_game_mapping(
        (row.game_title for row in filtered), config.processed_dir / "game_mapping.json",
    )
    train, test = leave_one_out(filtered, config.seed)
    counts = Counter(row.user_id for row in filtered)
    active_titles = {row.game_title for row in filtered}
    train_titles = {row.game_title for row in train}
    test_only_titles = {row.game_title for row in test} - train_titles
    stats = {
        "schema_version": 1,
        "source": {"dataset": "tamber/steam-video-games", "url": SOURCE_URL,
                   "filename": config.raw_path.name,
                   "sha256": hashlib.sha256(config.raw_path.read_bytes()).hexdigest()},
        "seed": config.seed,
        "min_games_per_user": config.min_games_per_user,
        "title_normalization": "NFKC, collapse whitespace, casefold",
        "duplicate_policy": "maximum positive play hours per normalized user/game pair",
        "raw": raw_stats,
        "filtering": {
            "n_users_before_min_games": len({row.user_id for row in interactions}),
            "n_users_dropped": len({row.user_id for row in interactions}) - len(counts),
            "n_interactions_dropped_for_min_games": len(interactions) - len(filtered),
        },
        "n_users": len(counts), "n_games": len(active_titles),
        "n_interactions": len(filtered), "mapping_size": len(mapping),
        "mapping_sha256": hashlib.sha256(json_text(mapping).encode("utf-8")).hexdigest(),
        "hours": describe(row.hours for row in filtered),
        "games_per_user": describe(counts.values()),
        "split": {
            "method": "random_leave_one_out",
            "rng": "random.Random with string seed '<seed>:<user_id>' over sorted titles",
            "n_train": len(train), "n_test": len(test),
            "n_train_games": len(train_titles),
            "n_test_games_unseen_in_train": len(test_only_titles),
            "n_test_users_with_unseen_game": sum(row.game_title in test_only_titles for row in test),
            "test_only_game_ids": sorted(mapping[title] for title in test_only_titles),
            "caveat": "No timestamps: random leave-one-out is not a temporal evaluation.",
        },
    }
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.processed_dir / "game_mapping.json", mapping)
    for filename, rows in (("interactions.csv", filtered), ("train.csv", train), ("test.csv", test)):
        write_csv(
            config.processed_dir / filename, ["user_id", "game_id", "game_title", "hours"],
            ((row.user_id, mapping[row.game_title], row.game_title, row.hours) for row in rows),
        )
    write_csv(
        config.processed_dir / "games.csv", ["game_id", "game_title", "display_title", "is_active"],
        ((game_id, title, display_titles.get(title, title), int(title in active_titles))
         for title, game_id in sorted(mapping.items(), key=lambda item: item[1])),
    )
    # Write stats last so its hashes describe a successfully completed preparation run.
    write_json(config.processed_dir / "stats.json", stats)
    LOGGER.info("Filtered: %s users | %s games | %s interactions", len(counts), len(active_titles), len(filtered))
    LOGGER.info("Split: %s train | %s test (one held-out game per user)", len(train), len(test))
    LOGGER.info("Hours: %s", json.dumps(stats["hours"], sort_keys=True))
    LOGGER.info("Merged %s duplicate play rows", raw_stats["n_duplicate_play_rows_merged"])
    if test_only_titles:
        LOGGER.warning(
            "%d test users have a positive absent from training (%d games); retained and reported, not resampled.",
            stats["split"]["n_test_users_with_unseen_game"], len(test_only_titles),
        )
    LOGGER.info("Persisted %s game IDs and all outputs to %s", len(mapping), config.processed_dir)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_pipeline(load_config(args.config))
    except (OSError, ValueError, csv.Error, yaml.YAMLError) as exc:
        parser.exit(1, f"Data pipeline failed: {exc}\n")


if __name__ == "__main__":
    main()
