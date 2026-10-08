"""Exercise the failure modes that could silently corrupt recommendation experiments."""

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from src.data import (
    Interaction, PipelineConfig, build_game_mapping, describe, filter_users,
    leave_one_out, load_config, load_interactions, normalize_title, run_pipeline,
)


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_raw(self, rows, name="raw.csv"):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)
        return path

    def sample_rows(self):
        rows = [
            [user, f"Game {game}", "play", game + 0.5, 0]
            for user in (101, 202, 303) for game in range(6)
        ]
        rows += [[404, f"Game {game}", "play", 1, 0] for game in range(4)]
        rows += [
            [101, "  GAME   0  ", "play", 10, 0],
            [101, "Purchase Only", "purchase", 1, 0],
            [101, "Zero Hours", "play", 0, 0],
        ]
        return rows

    def test_headerless_four_and_five_columns_preserve_first_row(self):
        for extra in ([], [0]):
            with self.subTest(extra=extra):
                path = self.write_raw([[123, "A game, with comma", "play", 1.5] + extra])
                rows, counts, _ = load_interactions(path)
                self.assertEqual(rows, [Interaction("123", "a game, with comma", 1.5)])
                self.assertEqual(counts["n_rows"], 1)

    def test_optional_header_and_large_user_id(self):
        path = self.write_raw([
            ["user-id", "game-title", "behavior-name", "value"],
            ["76561197970982479", "Portal", "play", 4],
        ])
        rows, counts, _ = load_interactions(path)
        self.assertEqual(rows[0].user_id, "76561197970982479")
        self.assertEqual(counts["n_rows"], 1)

    def test_normalization_is_idempotent_and_preserves_punctuation(self):
        title = "  ＰＯＲＴＡＬ\t  2: Test\u00a0Edition  "
        self.assertEqual(normalize_title(title), "portal 2: test edition")
        self.assertEqual(normalize_title(normalize_title(title)), normalize_title(title))

    def test_purchase_nonpositive_hours_and_duplicate_snapshots(self):
        path = self.write_raw([
            [1, " Portal ", "purchase", 1], [1, "Portal", "play", 3],
            [1, "PORTAL", "play", 5], [1, "portal", "play", 5],
            [1, "Zero", "play", 0], [1, "Negative", "play", -1],
        ])
        rows, counts, titles = load_interactions(path)
        self.assertEqual(rows, [Interaction("1", "portal", 5.0)])
        self.assertEqual(counts["n_purchase_rows_dropped"], 1)
        self.assertEqual(counts["n_nonpositive_play_rows_dropped"], 2)
        self.assertEqual(counts["n_duplicate_play_rows_merged"], 2)
        self.assertIn(titles["portal"], {"Portal", "PORTAL", "portal"})

    def test_invalid_rows_fail_instead_of_silently_changing_the_dataset(self):
        bad_rows = [
            [1, "Portal", "play", "nan"], [1, "Portal", "play", "inf"],
            [1, "Portal", "play", "unknown"], [1, "Portal", "other", 1],
            [1, " ", "play", 1], ["1.5", "Portal", "play", 1],
            [0, "Portal", "play", 1], [1, "Portal", "play"],
            [1, "Portal", "play", 1, 0, 0],
        ]
        for bad_row in bad_rows:
            with self.subTest(row=bad_row):
                with self.assertRaises(ValueError):
                    load_interactions(self.write_raw([bad_row]))

    def test_minimum_counts_distinct_games_after_deduplication(self):
        rows, _, _ = load_interactions(self.write_raw(self.sample_rows()))
        filtered = filter_users(rows, 5)
        self.assertEqual(len(filtered), 18)
        self.assertEqual({row.user_id for row in filtered}, {"101", "202", "303"})

    def test_mapping_sorted_initially_then_appends_without_renumbering(self):
        path = self.root / "mapping.json"
        original = build_game_mapping(["zulu", "bravo", "bravo"], path)
        self.assertEqual(original, {"bravo": 0, "zulu": 1})
        path.write_text(json.dumps(original), encoding="utf-8")
        updated = build_game_mapping(["alpha", "zulu"], path)
        self.assertEqual(updated, {"bravo": 0, "zulu": 1, "alpha": 2})

    def test_invalid_saved_mappings_are_rejected(self):
        path = self.root / "mapping.json"
        for mapping in (["portal"], {"Portal": 0}, {"portal": True},
                        {"portal": 1}, {"portal": 0, "dota": 0}):
            with self.subTest(mapping=mapping):
                path.write_text(json.dumps(mapping), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_game_mapping(["portal"], path)

    def test_split_is_order_independent_and_stable_for_unrelated_users(self):
        rows = [Interaction(str(user), f"game {game}", 1) for user in range(1, 11) for game in range(8)]
        first = leave_one_out(rows, seed=42)
        shuffled = rows.copy()
        random.Random(99).shuffle(shuffled)
        self.assertEqual(first, leave_one_out(shuffled, seed=42))
        subset = [row for row in rows if row.user_id != "1"]
        self.assertEqual([row for row in first[1] if row.user_id != "1"], leave_one_out(subset, 42)[1])
        self.assertNotEqual(first[1], leave_one_out(rows, seed=43)[1])

    def test_split_rejects_duplicate_pairs_and_single_game_users(self):
        row = Interaction("1", "portal", 10)
        with self.assertRaises(ValueError):
            leave_one_out([row, row], 42)
        with self.assertRaises(ValueError):
            leave_one_out([row], 42)

    def test_unseen_test_items_are_reported_without_resampling(self):
        raw = self.write_raw([[1, "Portal", "play", 4], [1, "Dota", "play", 8]])
        stats = run_pipeline(PipelineConfig(raw, self.root / "processed", min_games_per_user=2))
        self.assertEqual(stats["split"]["n_test_games_unseen_in_train"], 1)
        self.assertEqual(stats["split"]["n_test_users_with_unseen_game"], 1)
        self.assertEqual(stats["split"]["n_test"], 1)

    def test_config_paths_are_relative_to_project_and_reject_bad_values(self):
        directory = self.root / "config"
        directory.mkdir()
        config_path = directory / "default.yaml"
        config_path.write_text(
            "seed: 42\ndata:\n  raw_path: data/raw.csv\n  processed_dir: data/processed\n"
            "  min_games_per_user: 5\n", encoding="utf-8",
        )
        config = load_config(config_path)
        self.assertEqual(config.raw_path, self.root / "data/raw.csv")
        self.assertEqual(config.processed_dir, self.root / "data/processed")
        for minimum in (1, True, 5.5):
            with self.assertRaises(ValueError):
                PipelineConfig(config.raw_path, config.processed_dir, minimum)
        for seed in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                PipelineConfig(config.raw_path, config.processed_dir, seed=seed)

    def test_missing_or_empty_input_does_not_create_outputs(self):
        output = self.root / "processed"
        with self.assertRaises(FileNotFoundError):
            run_pipeline(PipelineConfig(self.root / "missing.csv", output))
        with self.assertRaises(ValueError):
            run_pipeline(PipelineConfig(self.write_raw([]), output))
        self.assertFalse(output.exists())

    def test_raw_input_cannot_be_overwritten_by_outputs(self):
        with self.assertRaises(ValueError):
            PipelineConfig(self.root / "train.csv", self.root)

    def test_hours_distribution(self):
        summary = describe([1, 2, 3, 4, 100])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 3)
        self.assertEqual(summary["p25"], 2)
        self.assertEqual(summary["p75"], 4)
        self.assertAlmostEqual(summary["mean"], 22)

    def test_complete_run_stats_and_byte_identical_rerun(self):
        raw = self.write_raw(self.sample_rows())
        output = self.root / "processed"
        config = PipelineConfig(raw, output)
        stats = run_pipeline(config)
        self.assertEqual((stats["n_users"], stats["n_games"], stats["n_interactions"]), (3, 6, 18))
        self.assertEqual((stats["split"]["n_train"], stats["split"]["n_test"]), (15, 3))
        self.assertEqual(stats["filtering"]["n_users_dropped"], 1)
        self.assertEqual(stats["filtering"]["n_interactions_dropped_for_min_games"], 4)
        self.assertEqual(json.loads((output / "stats.json").read_text()), stats)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(len(before), 6)
        run_pipeline(config)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        # Shuffling the CSV must preserve the mapping and splits; only source SHA changes.
        shuffled_rows = self.sample_rows()
        random.Random(12).shuffle(shuffled_rows)
        self.write_raw(shuffled_rows)
        run_pipeline(config)
        for name in before.keys() - {"stats.json"}:
            self.assertEqual(before[name], (output / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
