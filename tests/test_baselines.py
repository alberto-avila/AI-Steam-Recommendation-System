"""Numerical, leakage, persistence, and reproducibility checks for the baseline checkpoint."""

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from src.baselines import (
    ALSModel, BaselineConfig, PopularityModel, cases_for, confidence_matrix,
    fit_als, input_hashes, load_config, load_dataset, load_models, make_identity,
    popularity_counts, run_baselines, verify_artifacts,
)
from src.data import Interaction, PipelineConfig, leave_one_out, load_config as load_data_config, run_pipeline
from src.evaluate import evaluate_model, metrics_for_rank, prepare_cases, rank_games, summarize


class EvaluationTests(unittest.TestCase):
    def test_metric_boundaries_and_untruncated_mrr(self):
        self.assertEqual(metrics_for_rank(1), (1.0, 1.0))
        self.assertEqual(metrics_for_rank(10), (1.0, 0.1))
        self.assertEqual(metrics_for_rank(11), (0.0, 1 / 11))
        self.assertEqual(metrics_for_rank(None), (0.0, 0.0))
        for rank in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                metrics_for_rank(rank)

    def test_score_ties_use_game_ids_and_bad_scores_fail(self):
        np.testing.assert_array_equal(rank_games([8, 2, 7], [0.5, 0.5, 1.0]), [7, 2, 8])
        for ids, scores in [([1, 1], [1, 2]), ([1], [np.nan]), ([1], [np.inf]),
                            ([1], [1, 2]), ([-1], [0]), ([1.1], [0])]:
            with self.subTest(ids=ids, scores=scores), self.assertRaises(ValueError):
                rank_games(ids, scores)

    def test_fixed_candidates_exclude_known_positives_and_inactive_ids(self):
        counts = np.arange(1, 51, dtype=np.float32)
        counts[49] = 0
        known = {"u": {0, 1, 2}, "other": {3, 4}}
        first = prepare_cases({"u": 2}, counts, known, seed=42, n_negatives=10)[0]
        repeated = prepare_cases({"other": 4, "u": 2}, counts, known, seed=42, n_negatives=10)[1]
        self.assertTrue(first.supported)
        self.assertEqual(len(first.sampled_ids), 11)
        self.assertEqual(len(set(first.sampled_ids)), 11)
        self.assertIn(2, first.sampled_ids)
        self.assertFalse({0, 1, 49} & set(first.full_ids))
        self.assertFalse({0, 1, 49} & set(first.sampled_ids))
        np.testing.assert_array_equal(first.sampled_ids, repeated.sampled_ids)
        np.testing.assert_array_equal(first.full_ids, repeated.full_ids)
        changed = prepare_cases({"u": 2}, counts, known, seed=43, n_negatives=10)[0]
        self.assertFalse(np.array_equal(first.sampled_ids, changed.sampled_ids))

    def test_sampling_uses_popularity_to_the_configured_power(self):
        with patch("src.evaluate.user_rng") as generator:
            generator.return_value.choice.return_value = np.array([1])
            prepare_cases({"u": 0}, np.array([5, 1, 16]), {"u": {0}},
                          seed=42, n_negatives=1, exponent=0.75)
            arguments = generator.return_value.choice.call_args
            np.testing.assert_array_equal(arguments.args[0], [1, 2])
            np.testing.assert_allclose(arguments.kwargs["p"], [1 / 9, 8 / 9])
            self.assertFalse(arguments.kwargs["replace"])

    def test_small_negative_pool_and_unsupported_targets(self):
        cases = prepare_cases({"u": 0, "v": 3}, np.array([2, 1, 1, 0]),
                              {"u": {0, 1}, "v": {1, 3}}, seed=42)
        np.testing.assert_array_equal(cases[0].sampled_ids, [0, 2])
        self.assertFalse(cases[1].supported)
        self.assertNotIn(3, cases[1].sampled_ids)
        rows = evaluate_model(PopularityModel(np.array([2, 1, 1, 0])), cases, model_name="popularity", split="test")
        unsupported = [row for row in rows if row["user_id"] == "v"]
        self.assertTrue(all(row["rank"] is None and row["reciprocal_rank"] == 0 and row["recall_at_10"] == 0 for row in unsupported))
        summaries = summarize(rows)
        for row in summaries:
            self.assertEqual(row["coverage"], 0.5)
            self.assertEqual(row["n_users"], 2 if row["population"] == "all_users" else 1)
            self.assertEqual(row["recall_at_10"], 0.5 if row["population"] == "all_users" else 1)

    def test_empty_eligible_pool_reports_no_supported_metric(self):
        cases = prepare_cases({"u": 0}, np.zeros(3), {"u": {0}}, seed=42)
        self.assertEqual(len(cases[0].sampled_ids), 0)
        summaries = summarize(evaluate_model(PopularityModel(np.zeros(3)), cases, model_name="popularity", split="test"))
        for row in summaries:
            self.assertEqual(row["coverage"], 0)
            self.assertEqual(row["recall_at_10"], 0 if row["population"] == "all_users" else None)

    def test_sampler_validates_parameters(self):
        for counts, negatives, exponent in [(np.array([np.nan]), 1, 0.75),
                                            (np.array([-1]), 1, 0.75),
                                            (np.array([1]), 0, 0.75),
                                            (np.array([1]), 1, -1)]:
            with self.assertRaises(ValueError):
                prepare_cases({"u": 0}, counts, {"u": {0}}, seed=42,
                              n_negatives=negatives, exponent=exponent)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {"a": 0, "b": 1, "c": 2, "unseen": 3}
        self.users = {"10": 0, "20": 1}
        self.rows = [Interaction("10", "a", 3), Interaction("10", "b", 8),
                     Interaction("20", "a", 0.1), Interaction("20", "c", 4)]

    def test_popularity_counts_users_not_hours_or_duplicate_rows(self):
        counts = popularity_counts(self.rows + [self.rows[0]], self.mapping)
        np.testing.assert_array_equal(counts, [2, 1, 1, 0])
        np.testing.assert_array_equal(PopularityModel(counts).score("any-user", [3, 0]), [0, 2])

    def test_confidence_has_one_offset_and_one_alpha_and_fixed_indexing(self):
        matrix = confidence_matrix(self.rows, self.users, self.mapping, alpha=10)
        expected = [[1 + 10 * np.log(4), 1 + 10 * np.log(9), 0, 0],
                    [1 + 10 * np.log(1.1), 0, 1 + 10 * np.log(5), 0]]
        np.testing.assert_allclose(matrix.toarray(), expected, rtol=1e-6)
        self.assertEqual(matrix.dtype, np.float32)
        self.assertEqual(matrix.nnz, 4)
        with self.assertRaises(ValueError):
            confidence_matrix(self.rows + [self.rows[0]], self.users, self.mapping, alpha=10)

    def test_als_does_not_apply_alpha_twice(self):
        with patch("src.baselines.AlternatingLeastSquares") as constructor:
            constructor.return_value.user_factors = np.zeros((2, 2), dtype=np.float32)
            constructor.return_value.item_factors = np.zeros((4, 2), dtype=np.float32)
            fit_als(self.rows, self.users, self.mapping, factors=2, regularization=0.1,
                    alpha=40, iterations=2, seed=42)
            self.assertEqual(constructor.call_args.kwargs["alpha"], 1.0)
            self.assertEqual(constructor.call_args.kwargs["num_threads"], 1)
            matrix = constructor.return_value.fit.call_args.args[0]
            self.assertAlmostEqual(float(matrix[0, 0]), 1 + 40 * np.log(4), places=4)

    def test_als_scores_and_seeded_fits_are_reproducible(self):
        parameters = dict(factors=2, regularization=0.1, alpha=10, iterations=4, seed=42)
        first = fit_als(self.rows, self.users, self.mapping, **parameters)
        second = fit_als(self.rows, self.users, self.mapping, **parameters)
        np.testing.assert_array_equal(first.user_factors, second.user_factors)
        np.testing.assert_array_equal(first.item_factors, second.item_factors)
        with threadpool_limits(limits=1):
            expected = first.item_factors[[2, 0]] @ first.user_factors[self.users["10"]]
            np.testing.assert_array_equal(first.score("10", [2, 0]), expected)
        with self.assertRaises(KeyError):
            first.score("missing", [0])
        for bad_ids in ([-1], [4], [0.5]):
            with self.assertRaises(ValueError):
                first.score("10", bad_ids)


class BaselineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        raw = self.root / "raw" / "steam.csv"
        raw.parent.mkdir()
        with raw.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for user in range(1, 9):
                for game in range(6):
                    writer.writerow([user, f"game {(game + user) % 10}", "play", game + user + 0.1])
        self.data = PipelineConfig(raw, self.root / "processed")
        run_pipeline(self.data)
        self.config = BaselineConfig(
            self.data, self.root / "baselines", self.root / "results",
            factors=(2, 3), regularization=(0.1,), alpha=(10,), iterations=3, n_negatives=2,
        )

    def test_extended_and_old_configuration_remain_compatible(self):
        directory = self.root / "config"
        directory.mkdir()
        path = directory / "default.yaml"
        path.write_text(
            "seed: 42\ndata:\n  raw_path: raw/steam.csv\n  processed_dir: processed\n"
            "  min_games_per_user: 5\nbaselines:\n  factors: [2, 3]\nevaluation:\n  n_negatives: 2\n",
            encoding="utf-8",
        )
        self.assertEqual(load_data_config(path), self.data)
        config = load_config(path)
        self.assertEqual(config.factors, [2, 3])
        self.assertEqual(config.n_negatives, 2)
        with self.assertRaises(ValueError):
            replace(self.config, artifacts_dir=self.data.processed_dir)
        with self.assertRaises(ValueError):
            replace(self.config, factors=(2, 2))

    def test_validation_membership_is_disjoint_complete_and_keeps_test_out(self):
        dataset = load_dataset(self.data)
        fitting, validation = leave_one_out(dataset.train, self.data.seed + 1)
        pair_set = lambda rows: {(row.user_id, row.game_title) for row in rows}
        self.assertFalse(pair_set(fitting) & pair_set(validation))
        self.assertFalse((pair_set(fitting) | pair_set(validation)) & pair_set(dataset.test))
        self.assertCountEqual(fitting + validation, dataset.train)
        self.assertEqual(len(validation), len(dataset.users))
        counts = popularity_counts(fitting, dataset.mapping)
        cases = cases_for(validation, counts, dataset, self.config, "validation")
        for case in cases:
            competitors = set(case.full_ids) - {case.target_game_id}
            self.assertFalse(competitors & dataset.known_positives[case.user_id])

    def test_complete_run_roundtrip_cache_and_tampering_checks(self):
        original = input_hashes(self.data.processed_dir)
        metrics = run_baselines(self.config)
        self.assertEqual(input_hashes(self.data.processed_dir), original)
        self.assertEqual(len(metrics["validation_search"]), 2)
        self.assertEqual(sum(row["selected"] for row in metrics["validation_search"]), 1)
        self.assertEqual(len(metrics["summary"]), 16)
        identity = make_identity(self.config, original)
        verify_artifacts(self.config.artifacts_dir, identity)
        with patch("src.baselines.fit_als", side_effect=AssertionError("Valid cache should not refit")):
            self.assertEqual(run_baselines(self.config), metrics)
        popularity, als = load_models(self.config.artifacts_dir)
        dataset = load_dataset(self.data)
        cases = cases_for(dataset.test, popularity.counts, dataset, self.config, "test")
        replay = evaluate_model(als, cases, model_name="als", split="test")
        self.assertEqual(summarize(replay), [row for row in metrics["summary"] if row["split"] == "test" and row["model"] == "als"])
        self.assertTrue((self.config.results_dir / "comparison.md").is_file())
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            run_baselines(replace(self.config, iterations=4))
        candidate_path = self.config.artifacts_dir / "test_candidates.jsonl"
        with candidate_path.open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            load_models(self.config.artifacts_dir)

    def test_two_fresh_runs_have_identical_metrics_candidates_and_factors(self):
        first = run_baselines(self.config)
        other = replace(self.config, artifacts_dir=self.root / "other-baselines", results_dir=self.root / "other-results")
        second = run_baselines(other)
        self.assertEqual(first, second)
        for name in ("validation.csv", "validation_fit.csv", "test_candidates.jsonl", "validation_candidates.jsonl", "user_mapping.json", "per_user.csv"):
            self.assertEqual((self.config.artifacts_dir / name).read_bytes(), (other.artifacts_dir / name).read_bytes())
        _, first_als = load_models(self.config.artifacts_dir)
        _, second_als = load_models(other.artifacts_dir)
        np.testing.assert_array_equal(first_als.user_factors, second_als.user_factors)
        np.testing.assert_array_equal(first_als.item_factors, second_als.item_factors)

    def test_mismatched_game_mapping_and_overlap_are_rejected(self):
        path = self.data.processed_dir / "test.csv"
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines()
        fields = next(csv.reader([lines[1]]))
        fields[1] = "99999"
        lines[1] = ",".join(fields)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_dataset(self.data)
        path.write_text(original, encoding="utf-8")
        train = self.data.processed_dir / "train.csv"
        with train.open("a", encoding="utf-8") as handle:
            handle.write(original.splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_dataset(self.data)


if __name__ == "__main__":
    unittest.main()
