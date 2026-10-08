"""Shared candidate generation and ranking metrics for baselines and future models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
from threadpoolctl import threadpool_limits


class Scorer(Protocol):
    def score(self, user_id: str, game_ids: Sequence[int]) -> np.ndarray:
        """Return one finite score per requested game, in the same order."""
        ...


@dataclass(frozen=True)
class EvaluationCase:
    user_id: str
    target_game_id: int
    supported: bool
    full_ids: np.ndarray
    sampled_ids: np.ndarray

    def sampled_record(self) -> dict:
        return {
            "user_id": self.user_id, "target_game_id": self.target_game_id,
            "supported": self.supported, "game_ids": self.sampled_ids.tolist(),
        }


def user_rng(seed: int, namespace: str, user_id: str) -> np.random.Generator:
    # Separate streams avoid coupling one user's candidates to another user's history.
    digest = hashlib.sha256(f"{seed}:{namespace}:{user_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def prepare_cases(
    targets: Mapping[str, int], popularity: np.ndarray,
    known_positives: Mapping[str, set[int]], *, seed: int,
    n_negatives: int = 100, exponent: float = 0.75, namespace: str = "test",
) -> list[EvaluationCase]:
    """Use fit-only popularity; exclude ALL known positives from the negative pool.

    Other holdouts are used only as exclusion masks, never as fitting observations.
    Unsupported targets remain in case metadata but cannot enter the eligible ranking.
    """
    popularity = np.asarray(popularity)
    if popularity.ndim != 1 or not np.isfinite(popularity).all() or (popularity < 0).any():
        raise ValueError("Popularity must be a finite, nonnegative vector.")
    if type(n_negatives) is not int or n_negatives < 1:
        raise ValueError("n_negatives must be a positive integer.")
    if not np.isfinite(exponent) or exponent < 0:
        raise ValueError("Popularity exponent must be finite and nonnegative.")
    eligible = np.flatnonzero(popularity > 0)
    cases = []
    for user_id, target in sorted(targets.items()):
        if target not in known_positives.get(user_id, set()) or not 0 <= target < len(popularity):
            raise ValueError(f"Target for user {user_id} is missing from the known catalog/history.")
        pool = np.array([game for game in eligible if game not in known_positives[user_id]], dtype=np.int64)
        size = min(n_negatives, len(pool))
        if size == len(pool):
            negatives = pool.copy()
        else:
            weights = popularity[pool].astype(np.float64) ** exponent
            negatives = user_rng(seed, namespace, user_id).choice(
                pool, size=size, replace=False, p=weights / weights.sum(),
            )
        supported = bool(popularity[target] > 0)
        positive = np.array([target] if supported else [], dtype=np.int64)
        cases.append(EvaluationCase(
            user_id=user_id, target_game_id=int(target), supported=supported,
            full_ids=np.sort(np.concatenate((pool, positive))),
            sampled_ids=np.sort(np.concatenate((negatives, positive))),
        ))
    return cases


def rank_games(game_ids: Sequence[int], scores: Sequence[float]) -> np.ndarray:
    """Higher score wins; exact ties use ascending persisted game ID for every model."""
    ids = np.asarray(game_ids)
    values = np.asarray(scores)
    if ids.ndim != 1 or values.shape != ids.shape:
        raise ValueError("Game IDs and scores must be matching one-dimensional arrays.")
    if len(ids) and (not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any()):
        raise ValueError("Candidate IDs must be nonnegative integers.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Candidates must be unique.")
    if not np.isfinite(values).all():
        raise ValueError("Model returned nonfinite scores.")
    return ids[np.lexsort((ids, -values.astype(np.float64)))]


def metrics_for_rank(rank: int | None) -> tuple[float, float]:
    if rank is None:
        return 0.0, 0.0
    if type(rank) is not int or rank < 1:
        raise ValueError("Rank must be a positive integer or None for an unavailable target.")
    return float(rank <= 10), 1.0 / rank  # MRR is deliberately not truncated at ten.


@threadpool_limits.wrap(limits=1)
def evaluate_model(
    model: Scorer, cases: list[EvaluationCase], *, model_name: str,
    split: str, protocols: Sequence[str] = ("full_catalog", "sampled"),
) -> list[dict]:
    records = []
    for protocol in protocols:
        if protocol not in {"full_catalog", "sampled"}:
            raise ValueError(f"Unknown evaluation protocol: {protocol}")
        for case in cases:
            candidates = case.full_ids if protocol == "full_catalog" else case.sampled_ids
            rank = None
            if case.supported:
                ordered = rank_games(candidates, model.score(case.user_id, candidates))
                position = np.flatnonzero(ordered == case.target_game_id)
                if len(position) != 1:
                    raise ValueError("A supported target must appear exactly once in its candidates.")
                rank = int(position[0]) + 1
            recall, reciprocal = metrics_for_rank(rank)
            records.append({
                "model": model_name, "split": split, "protocol": protocol,
                "user_id": case.user_id, "target_game_id": case.target_game_id,
                "supported": case.supported, "n_candidates": len(candidates),
                "rank": rank, "recall_at_10": recall, "reciprocal_rank": reciprocal,
            })
    return records


def summarize(records: list[dict]) -> list[dict]:
    """Keep protocol and cohort denominators explicit; empty cohorts have null metrics."""
    summaries = []
    groups = sorted({(row["split"], row["model"], row["protocol"]) for row in records})
    for split, model, protocol in groups:
        group = [row for row in records if (row["split"], row["model"], row["protocol"]) == (split, model, protocol)]
        supported = [row for row in group if row["supported"]]
        for population, rows in (("all_users", group), ("supported_users", supported)):
            summaries.append({
                "split": split, "model": model, "protocol": protocol, "population": population,
                "n_users": len(rows), "n_supported": len(supported), "n_total": len(group),
                "coverage": len(supported) / len(group),
                "recall_at_10": sum(row["recall_at_10"] for row in rows) / len(rows) if rows else None,
                "mrr": sum(row["reciprocal_rank"] for row in rows) / len(rows) if rows else None,
            })
    return summaries
