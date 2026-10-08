"""Train-only popularity, validation-tuned CPU ALS, and a reproducible evaluation run.

Run ``python -m src.baselines --config config/default.yaml``. Install ``.[baselines]``
first. The six original processed artifacts are read-only inputs to this checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
import platform
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_info, threadpool_limits
import yaml

from src.data import (
    ARTIFACT_NAMES, Interaction, PipelineConfig, build_game_mapping, json_text,
    leave_one_out, load_config as load_data_config, write_csv, write_json,
)
from src.evaluate import EvaluationCase, evaluate_model, prepare_cases, summarize

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
CACHE_FILES = {
    "user_mapping.json", "validation_fit.csv", "validation.csv",
    "validation_candidates.jsonl", "test_candidates.jsonl", "validation_popularity.npy",
    "popularity.npy", "als.npz", "validation_search.csv", "per_user.csv", "metrics.json",
}


@dataclass(frozen=True)
class BaselineConfig:
    data: PipelineConfig
    artifacts_dir: Path
    results_dir: Path
    factors: tuple[int, ...] = (32, 64)
    regularization: tuple[float, ...] = (0.01, 0.1)
    alpha: tuple[float, ...] = (10.0, 40.0)
    iterations: int = 30
    n_negatives: int = 100
    popularity_exponent: float = 0.75

    def __post_init__(self) -> None:
        for name in ("factors", "regularization", "alpha"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be a nonempty sequence without duplicates.")
            if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{name} must contain finite positive numbers.")
        if any(type(value) is not int for value in self.factors):
            raise ValueError("factors must contain integers.")
        for name in ("iterations", "n_negatives"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.popularity_exponent) not in (int, float) or not math.isfinite(self.popularity_exponent) or self.popularity_exponent < 0:
            raise ValueError("popularity_exponent must be finite and nonnegative.")
        outputs = [self.artifacts_dir.resolve(), self.results_dir.resolve()]
        inputs = [self.data.processed_dir.resolve(), self.data.raw_path.resolve().parent]
        # Keep generated outputs apart from immutable data and from each other.
        for left, right in itertools.combinations(outputs, 2):
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Baseline artifacts and report directories must not overlap.")
        for output, source in itertools.product(outputs, inputs):
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError("Baseline outputs must not overlap raw or processed data directories.")

    def settings(self) -> dict:
        return {
            "seed": self.data.seed, "validation_seed": self.data.seed + 1,
            "min_games_per_user": self.data.min_games_per_user,
            "factors": list(self.factors), "regularization": list(self.regularization),
            "alpha": list(self.alpha), "iterations": self.iterations,
            "n_negatives": self.n_negatives, "popularity_exponent": self.popularity_exponent,
            "dtype": "float32", "num_threads": 1, "library_alpha": 1.0,
            "confidence": "1 + alpha * log1p(hours)",
            "selection": "full_catalog/all_users Recall@10, then MRR, then grid order",
            "ties": "score descending, game_id ascending",
            "unsupported_targets": "zero Recall@10 and reciprocal rank; null rank",
            "candidate_rng": "PCG64; first 16 SHA256 bytes of seed:split:user_id",
        }


def load_config(path: Path) -> BaselineConfig:
    data = load_data_config(path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    baseline = document.get("baselines", {})
    evaluation = document.get("evaluation", {})
    if set(baseline) - {"artifacts_dir", "results_dir", "factors", "regularization", "alpha", "iterations"}:
        raise ValueError("Unknown baseline configuration key.")
    if set(evaluation) - {"n_negatives", "popularity_exponent"}:
        raise ValueError("Unknown evaluation configuration key.")
    root = path.resolve().parent.parent
    paths = {}
    for key, default in (("artifacts_dir", "data/baselines"), ("results_dir", "results")):
        value = baseline.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string.")
        paths[key] = root / value
    return BaselineConfig(
        data=data, **paths,
        **{key: value for key, value in baseline.items() if key not in paths},
        **evaluation,
    )


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(directory: Path) -> dict[str, str]:
    return {name: file_hash(directory / name) for name in ARTIFACT_NAMES}


def read_interactions(path: Path, mapping: dict[str, int]) -> list[Interaction]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "game_id", "game_title", "hours"]:
            raise ValueError(f"Unexpected interaction schema: {path.name}")
        for raw in reader:
            user, title, hours = raw["user_id"], raw["game_title"], float(raw["hours"])
            if not user.isascii() or not user.isdecimal() or int(user) <= 0 or user != str(int(user)):
                raise ValueError(f"Invalid user ID in {path.name}")
            if title not in mapping or int(raw["game_id"]) != mapping[title]:
                raise ValueError(f"Game ID/title mismatch in {path.name}")
            if not math.isfinite(hours) or hours <= 0 or (user, title) in seen:
                raise ValueError(f"Nonpositive/nonfinite hours or duplicate interaction in {path.name}")
            seen.add((user, title))
            rows.append(Interaction(user, title, hours))
    if not rows:
        raise ValueError(f"No interactions in {path.name}")
    return sorted(rows, key=lambda row: (row.user_id, row.game_title))


@dataclass
class Dataset:
    train: list[Interaction]
    test: list[Interaction]
    mapping: dict[str, int]
    users: dict[str, int]
    known_positives: dict[str, set[int]]


def load_dataset(config: PipelineConfig) -> Dataset:
    directory = config.processed_dir
    mapping = build_game_mapping([], directory / "game_mapping.json")
    train = read_interactions(directory / "train.csv", mapping)
    test = read_interactions(directory / "test.csv", mapping)
    interactions = read_interactions(directory / "interactions.csv", mapping)
    train_pairs = {(row.user_id, row.game_title) for row in train}
    test_pairs = {(row.user_id, row.game_title) for row in test}
    if train_pairs & test_pairs or set(train + test) != set(interactions):
        raise ValueError("Original splits overlap or do not preserve the complete interactions and hours.")
    train_counts, test_counts = Counter(row.user_id for row in train), Counter(row.user_id for row in test)
    if train_counts.keys() != test_counts.keys() or any(count != 1 for count in test_counts.values()):
        raise ValueError("Expected exactly one test positive for every training user.")
    if min(train_counts.values()) < max(2, config.min_games_per_user - 1):
        raise ValueError("Insufficient original training history for the validation holdout.")
    stats = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
    if stats["seed"] != config.seed or stats["min_games_per_user"] != config.min_games_per_user:
        raise ValueError("Data configuration differs from the existing processed split.")
    if stats["mapping_sha256"] != file_hash(directory / "game_mapping.json"):
        raise ValueError("Persisted mapping differs from the data pipeline's recorded mapping.")
    if (stats["n_interactions"], stats["n_users"], stats["n_games"], stats["mapping_size"]) != (
        len(interactions), len(train_counts), len({row.game_title for row in interactions}), len(mapping),
    ) or (stats["split"]["n_train"], stats["split"]["n_test"]) != (len(train), len(test)):
        raise ValueError("Processed counts do not match stats.json.")
    known: dict[str, set[int]] = defaultdict(set)
    for row in interactions:
        known[row.user_id].add(mapping[row.game_title])
    users = {user: index for index, user in enumerate(sorted(train_counts))}
    return Dataset(train, test, mapping, users, dict(known))


def popularity_counts(rows: list[Interaction], mapping: dict[str, int]) -> np.ndarray:
    counts = np.zeros(len(mapping), dtype=np.float32)
    for _, title in {(row.user_id, row.game_title) for row in rows}:
        counts[mapping[title]] += 1
    return counts


def confidence_matrix(
    rows: list[Interaction], users: dict[str, int], mapping: dict[str, int], alpha: float,
) -> csr_matrix:
    pairs = [(row.user_id, row.game_title) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Confidence matrix requires unique user/game pairs.")
    hours = np.array([row.hours for row in rows], dtype=np.float64)
    if not np.isfinite(hours).all() or (hours <= 0).any() or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Hours must be finite and positive; alpha must be finite and nonnegative.")
    # Missing entries retain implicit preference=0/confidence=1 in the ALS solver.
    values = (1 + alpha * np.log1p(hours)).astype(np.float32)
    matrix = csr_matrix((values, (
        [users[row.user_id] for row in rows], [mapping[row.game_title] for row in rows],
    )), shape=(len(users), len(mapping)), dtype=np.float32)
    matrix.sort_indices()
    return matrix


def checked_ids(game_ids: Sequence[int], size: int) -> np.ndarray:
    ids = np.asarray(game_ids)
    if ids.ndim != 1 or (len(ids) and (
        not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids >= size).any()
    )):
        raise ValueError("Game IDs must be valid integer mapping indices.")
    return ids.astype(np.int64, copy=False)


@dataclass
class PopularityModel:
    counts: np.ndarray

    def score(self, user_id: str, game_ids: Sequence[int]) -> np.ndarray:
        return self.counts[checked_ids(game_ids, len(self.counts))]


@dataclass
class ALSModel:
    user_factors: np.ndarray
    item_factors: np.ndarray
    users: dict[str, int]

    def score(self, user_id: str, game_ids: Sequence[int]) -> np.ndarray:
        return self.item_factors[checked_ids(game_ids, len(self.item_factors))] @ self.user_factors[self.users[user_id]]


@threadpool_limits.wrap(limits=1)
def fit_als(
    rows: list[Interaction], users: dict[str, int], mapping: dict[str, int], *,
    factors: int, regularization: float, alpha: float, iterations: int, seed: int,
) -> ALSModel:
    matrix = confidence_matrix(rows, users, mapping, alpha)
    model = AlternatingLeastSquares(
        factors=factors, regularization=regularization, alpha=1.0,
        dtype=np.float32, iterations=iterations, num_threads=1, random_state=seed,
    )
    model.fit(matrix, show_progress=False)
    if not np.isfinite(model.user_factors).all() or not np.isfinite(model.item_factors).all():
        raise ValueError("ALS produced nonfinite factors.")
    return ALSModel(model.user_factors, model.item_factors, users)


def cases_for(rows: list[Interaction], counts: np.ndarray, dataset: Dataset,
              config: BaselineConfig, split: str) -> list[EvaluationCase]:
    return prepare_cases(
        {row.user_id: dataset.mapping[row.game_title] for row in rows}, counts,
        dataset.known_positives, seed=config.data.seed, n_negatives=config.n_negatives,
        exponent=config.popularity_exponent, namespace=split,
    )


@threadpool_limits.wrap(limits=1)
def make_identity(config: BaselineConfig, hashes: dict[str, str]) -> dict:
    code = Path(__file__).resolve().parent
    return {
        "schema_version": SCHEMA_VERSION, "input_sha256": hashes,
        "configuration": config.settings(),
        "code_sha256": {name: file_hash(code / name) for name in ("data.py", "baselines.py", "evaluate.py")},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "implicit", "threadpoolctl", "PyYAML")},
        "python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
        "numerical_libraries": [
            {key: library.get(key) for key in ("internal_api", "version", "architecture", "num_threads")}
            for library in threadpool_info()
        ],
    }


def verify_artifacts(directory: Path, expected_identity: dict | None = None) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Baseline artifacts are incomplete: manifest.json is missing. Use a new artifacts_dir.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported baseline artifact version.")
    if expected_identity is not None and manifest.get("identity") != expected_identity:
        raise ValueError("Incompatible cached baseline artifacts: input, configuration, code, or environment changed. Use a new artifacts_dir.")
    if set(manifest.get("artifact_sha256", {})) != CACHE_FILES:
        raise ValueError("Baseline artifact inventory is incomplete or unexpected.")
    for name, expected in manifest["artifact_sha256"].items():
        if not (directory / name).is_file() or file_hash(directory / name) != expected:
            raise ValueError(f"Baseline artifact is missing or modified: {name}")
    return manifest


def load_models(directory: Path) -> tuple[PopularityModel, ALSModel]:
    """Load verified, pickle-free inference artifacts; callers can also verify run identity."""
    verify_artifacts(directory)
    users = json.loads((directory / "user_mapping.json").read_text(encoding="utf-8"))
    if sorted(users.values()) != list(range(len(users))):
        raise ValueError("Invalid persisted user mapping.")
    counts = np.load(directory / "popularity.npy", allow_pickle=False)
    with np.load(directory / "als.npz", allow_pickle=False) as arrays:
        user_factors, item_factors = arrays["user_factors"], arrays["item_factors"]
    if user_factors.shape[0] != len(users) or item_factors.shape[0] != len(counts) or user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("Model shapes do not match the persisted ID mappings.")
    return PopularityModel(counts), ALSModel(user_factors, item_factors, users)


def write_rows(path: Path, rows: list[dict]) -> None:
    header = list(rows[0])
    write_csv(path, header, ([row[key] for key in header] for row in rows))


def write_interactions(path: Path, rows: list[Interaction], mapping: dict[str, int]) -> None:
    write_csv(path, ["user_id", "game_id", "game_title", "hours"], (
        (row.user_id, mapping[row.game_title], row.game_title, row.hours) for row in rows
    ))


def write_candidates(path: Path, cases: list[EvaluationCase]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case.sampled_record(), sort_keys=True) + "\n")


def select_summary(records: list[dict], protocol: str = "full_catalog") -> dict:
    return next(row for row in summarize(records) if row["protocol"] == protocol and row["population"] == "all_users")


def write_report(config: BaselineConfig, metrics: dict, identity: dict) -> None:
    lines = [
        "# Popularity and tuned ALS baselines", "",
        "Both models use the same saved split and candidates. Test scores were evaluated only after validation selected ALS settings.", "",
        f"Seed: **{config.data.seed}**. Selected ALS: **{metrics['selected_als']}**.", "",
        "Full-catalog validation Recall@10 selects the winner; ties use MRR, then the fixed grid order.", "",
        "| Evaluation | Population | Model | Users | Recall@10 | MRR |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for protocol in ("full_catalog", "sampled"):
        for population in ("all_users", "supported_users"):
            for model in ("popularity", "als"):
                row = next(item for item in metrics["summary"] if (
                    item["split"], item["protocol"], item["population"], item["model"]
                ) == ("test", protocol, population, model))
                recall = f"{row['recall_at_10']:.6f}" if row["recall_at_10"] is not None else "N/A"
                mrr = f"{row['mrr']:.6f}" if row["mrr"] is not None else "N/A"
                lines.append(f"| {protocol} | {population} | {model} | {row['n_users']:,} | {recall} | {mrr} |")
    coverage = next(row for row in metrics["summary"] if row["split"] == "test")
    lines += [
        "", f"Coverage: **{coverage['n_supported']:,}/{coverage['n_total']:,} ({coverage['coverage']:.2%})** test targets have training support.",
        "Unsupported targets stay in all-user metrics with Recall@10=0 and reciprocal rank=0; their rank is unavailable, not last place.",
        "", "## Protocol", "",
        "- Full catalog means games represented in fitting data, excluding all of the user's known positives except the evaluated target.",
        f"- Sampled evaluation uses up to {config.n_negatives} unique alternatives weighted by fit-only popularity^{config.popularity_exponent}; candidates are fixed across models.",
        "- Other holdout identities are used only to exclude known positives from candidate competitors; their hours and scores are not used to fit or select models.",
        "- Score ties use ascending game ID. MRR uses the complete ranking, with no cutoff at ten.",
        "- These are two different candidate universes: sampled scores must not be presented as full-catalog scores.",
        "", "## Validation search", "",
        "| Grid order | Factors | Regularization | Alpha | Recall@10 | MRR | Selected |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trial in metrics["validation_search"]:
        lines.append(f"| {trial['grid_order']} | {trial['factors']} | {trial['regularization']} | {trial['alpha']} | {trial['recall_at_10']:.6f} | {trial['mrr']:.6f} | {trial['selected']} |")
    test_rows = {row["model"]: row for row in metrics["summary"] if (row["split"], row["protocol"], row["population"]) == ("test", "full_catalog", "all_users")}
    difference = test_rows["als"]["recall_at_10"] - test_rows["popularity"]["recall_at_10"]
    lines += [
        "", "## Interpretation and limits", "",
        f"ALS's full-catalog all-user Recall@10 differs from popularity by **{100 * difference:+.2f} percentage points**.",
        "A higher sampled score alone does not establish strong catalog-wide recommendations. No LLM result has been produced at this checkpoint.",
        "Steam-200k has no timestamps: this random evaluation is not future-play prediction. Hours are noisy engagement proxies; the catalog is historical, short-history users are excluded, and no online evaluation is available.",
        "", "## Reproducibility", "",
        "Run `python -m src.baselines --config config/default.yaml`. Identical completed artifacts are reused; incompatible or modified artifacts are rejected.",
        f"Dependencies: `{json.dumps(identity['versions'], sort_keys=True)}`.",
        f"Run identity SHA-256: `{hashlib.sha256(json_text(identity).encode('utf-8')).hexdigest()}`.",
        "Machine-readable aggregate results are in `metrics.json`; model files, candidate sets, validation membership, and per-user ranks are in the configured baseline artifacts directory.", "",
    ]
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(config.results_dir / "metrics.json", metrics)


@threadpool_limits.wrap(limits=1)
def run_baselines(config: BaselineConfig) -> dict:
    hashes = input_hashes(config.data.processed_dir)
    dataset = load_dataset(config.data)
    identity = make_identity(config, hashes)
    if config.artifacts_dir.exists():
        verify_artifacts(config.artifacts_dir, identity)
        metrics = json.loads((config.artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
        write_report(config, metrics, identity)
        LOGGER.info("Reused verified baseline artifacts: %s", config.artifacts_dir)
        return metrics

    fit_rows, validation = leave_one_out(dataset.train, config.data.seed + 1)
    fit_counts = popularity_counts(fit_rows, dataset.mapping)
    validation_cases = cases_for(validation, fit_counts, dataset, config, "validation")
    LOGGER.info("Validation: %d fit rows, %d held-out users, %d supported targets",
                len(fit_rows), len(validation), sum(case.supported for case in validation_cases))
    trials, best_model, best_key, selected = [], None, (-1.0, -1.0), None
    grid = list(itertools.product(config.factors, config.regularization, config.alpha))
    for order, (factors, regularization, alpha) in enumerate(grid):
        parameters = {"factors": factors, "regularization": regularization, "alpha": alpha}
        LOGGER.info("ALS search %d/%d: %s", order + 1, len(grid), parameters)
        model = fit_als(fit_rows, dataset.users, dataset.mapping, **parameters,
                        iterations=config.iterations, seed=config.data.seed)
        validation_records = evaluate_model(model, validation_cases, model_name="als", split="validation", protocols=("full_catalog",))
        summary = select_summary(validation_records)
        key = (summary["recall_at_10"], summary["mrr"])
        trials.append({"grid_order": order, **parameters, "recall_at_10": key[0], "mrr": key[1], "selected": False})
        LOGGER.info("Validation Recall@10=%.6f MRR=%.6f", *key)
        if key > best_key:  # Strict comparison preserves fixed grid order on exact ties.
            best_key, best_model, selected = key, model, parameters
    selected_order = next(index for index, trial in enumerate(trials) if all(trial[key] == value for key, value in selected.items()))
    trials[selected_order]["selected"] = True
    records = evaluate_model(best_model, validation_cases, model_name="als", split="validation")
    records += evaluate_model(PopularityModel(fit_counts), validation_cases, model_name="popularity", split="validation")

    # Only now refit on all original training rows and evaluate final test scores.
    LOGGER.info("Selected %s; refitting on all %d original training rows", selected, len(dataset.train))
    final_counts = popularity_counts(dataset.train, dataset.mapping)
    final_model = fit_als(dataset.train, dataset.users, dataset.mapping, **selected,
                          iterations=config.iterations, seed=config.data.seed)
    test_cases = cases_for(dataset.test, final_counts, dataset, config, "test")
    records += evaluate_model(final_model, test_cases, model_name="als", split="test")
    records += evaluate_model(PopularityModel(final_counts), test_cases, model_name="popularity", split="test")
    metrics = {
        "selected_als": {**selected, "iterations": config.iterations},
        "validation_search": trials, "summary": summarize(records),
        "configuration": config.settings(),
        "counts": {"validation_fit": len(fit_rows), "validation": len(validation), "train": len(dataset.train), "test": len(dataset.test)},
    }
    if input_hashes(config.data.processed_dir) != hashes:
        raise ValueError("Processed input files changed during baseline execution.")

    config.artifacts_dir.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, verified directory; failed fitting leaves no reusable cache.
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=config.artifacts_dir.parent) as temporary:
        staging = Path(temporary)
        if not staging.resolve().is_relative_to(config.artifacts_dir.parent.resolve()):
            raise ValueError("Unexpected staging directory.")
        write_json(staging / "user_mapping.json", dataset.users)
        write_interactions(staging / "validation_fit.csv", fit_rows, dataset.mapping)
        write_interactions(staging / "validation.csv", validation, dataset.mapping)
        write_candidates(staging / "validation_candidates.jsonl", validation_cases)
        write_candidates(staging / "test_candidates.jsonl", test_cases)
        np.save(staging / "validation_popularity.npy", fit_counts, allow_pickle=False)
        np.save(staging / "popularity.npy", final_counts, allow_pickle=False)
        np.savez(staging / "als.npz", user_factors=final_model.user_factors, item_factors=final_model.item_factors)
        write_rows(staging / "validation_search.csv", trials)
        write_rows(staging / "per_user.csv", records)
        write_json(staging / "metrics.json", metrics)
        write_json(staging / "manifest.json", {
            "schema_version": SCHEMA_VERSION, "identity": identity,
            "artifact_sha256": {name: file_hash(staging / name) for name in sorted(CACHE_FILES)},
        })
        verify_artifacts(staging, identity)
        staging.rename(config.artifacts_dir)
    write_report(config, metrics, identity)
    for row in metrics["summary"]:
        if row["split"] == "test" and row["population"] == "all_users":
            LOGGER.info("%s / %s: Recall@10=%.6f MRR=%.6f coverage=%.2f%%",
                        row["model"], row["protocol"], row["recall_at_10"], row["mrr"], 100 * row["coverage"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_baselines(load_config(args.config))
    except (OSError, ValueError, csv.Error, yaml.YAMLError) as exc:
        parser.exit(1, f"Baseline run failed: {exc}\n")


if __name__ == "__main__":
    main()
