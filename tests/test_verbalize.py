"""Hand-calculated rendering, token contracts, isolation, and reproducibility checks."""

import dataclasses
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from tokenizers import Tokenizer, models, pre_tokenizers

from src.data import Interaction, PipelineConfig, leave_one_out, run_pipeline, write_csv, write_json
from src.verbalize import (
    GemmaTokenizer, build_records, file_hash, format_hours, load_config, load_sources,
    make_identity, prepare_tokenizer, render_history, run_verbalizer, select_samples,
    summarize_records, verify_artifacts,
)


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.mapping = {"portal": 0, "portal 2": 1, "dota 2": 2, "short session": 3}
        self.display = {0: "Portal", 1: "Portal 2", 2: "Dota 2", 3: "Short Session"}
        self.rows = [Interaction("1", "portal", 999), Interaction("1", "portal 2", 20),
                     Interaction("1", "dota 2", 20), Interaction("1", "short session", .1)]

    def render(self, rows=None, **kwargs):
        return render_history(self.rows if rows is None else rows, self.mapping, self.display,
                              excluded_game_ids={0}, **kwargs)

    def test_exclude_before_top_k_and_break_hours_ties_by_persistent_id(self):
        result = self.render(k=2)
        self.assertEqual(result.text, "This player's most-played games:\nPortal 2 (20 hrs), Dota 2 (20 hrs).")
        self.assertEqual(result.game_ids, (1, 2))
        self.assertEqual((result.source_games, result.excluded_games, result.omitted_by_k), (4, 1, 1))
        shuffled = list(self.rows)
        random.Random(42).shuffle(shuffled)
        self.assertEqual(self.render(shuffled, k=2), result)

    def test_identity_exclusion_keeps_different_game_with_shared_title_prefix(self):
        self.assertEqual(self.render().game_ids, (1, 2, 3))
        self.assertIn("Portal 2", self.render().text)
        self.assertNotIn("Portal (", self.render().text)

    def test_normalized_identity_blocks_casing_unicode_whitespace_bypass(self):
        rows = [dataclasses.replace(self.rows[0], game_title="  ＰＯＲＴＡＬ  "), *self.rows[1:]]
        self.assertEqual(self.render(rows), self.render())

    def test_hours_are_not_rounded_and_integer_trailing_zero_is_preserved(self):
        self.assertEqual([format_hours(value) for value in [.1, 1.0, 10.0, 100.25, 10442]],
                         ["0.1", "1", "10", "100.25", "10442"])
        self.assertIn("Short Session (0.1 hrs)", self.render().text)

    def test_summary_strategy_reports_post_exclusion_breadth_volume_and_concentration(self):
        # Kept after excluding Portal: 20 + 20 + 0.1 = 40.1 hrs; top share 20/40.1 = 49.9%.
        self.assertEqual(self.render(k=2, strategy="top_hours_summary").text,
                         "This player has played 3 games for 40.1 hrs in total.\n"
                         "Their most-played game accounts for 50% of their playtime.\n"
                         "Most-played games (top 2 of 3):\n"
                         "Portal 2 (20 hrs), Dota 2 (20 hrs).")
        uncapped = self.render(strategy="top_hours_summary")
        self.assertIn("\nMost-played games:\n", uncapped.text)
        self.assertEqual(uncapped.game_ids, self.render().game_ids)

    def test_summary_concentration_never_rounds_to_misleading_extremes(self):
        dominant = [Interaction("1", "portal 2", 999), Interaction("1", "short session", .1)]
        self.assertIn("accounts for 99% of", self.render(dominant, strategy="top_hours_summary").text)
        self.mapping.update({f"game {index}": 10 + index for index in range(300)})
        self.display.update({10 + index: f"Game {index}" for index in range(300)})
        spread = [Interaction("1", f"game {index}", 1) for index in range(300)]
        self.assertIn("accounts for under 1% of", self.render(spread, strategy="top_hours_summary").text)

    def test_invalid_histories_and_parameters_fail(self):
        for rows in ([], [self.rows[0]], self.rows + [self.rows[1]],
                     [dataclasses.replace(self.rows[1], hours=float("nan"))],
                     [dataclasses.replace(self.rows[1], hours=0)],
                     [dataclasses.replace(self.rows[1], game_title="unknown")],
                     [self.rows[1], dataclasses.replace(self.rows[2], user_id="2")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.render(rows)
        for options in ({"k": 0}, {"k": True}, {"strategy": "unknown"}):
            with self.assertRaises(ValueError):
                self.render(**options)
        with self.assertRaises(ValueError):
            render_history(self.rows, self.mapping, {**self.display, 2: "Wrong game"}, excluded_game_ids={0})

    def test_sample_selection_is_unique_stable_and_covers_extremes(self):
        rows = [{"user_id": str(index), "history_games": index // 3} for index in range(50)]
        samples = select_samples(rows, 10, 42)
        self.assertEqual(samples, select_samples(list(reversed(rows)), 10, 42))
        self.assertEqual(len({row["user_id"] for row in samples}), 10)
        self.assertEqual([samples[0]["history_games"], samples[-1]["history_games"]], [0, 16])
        self.assertEqual(len(select_samples(rows[:3], 10, 42)), 3)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "config").mkdir()
        raw = self.root / "data/raw/steam.csv"
        raw.parent.mkdir(parents=True)
        raw.write_text("".join(f"{user},Game {game},play,{game + user / 10}\n"
                               for user in range(1, 4) for game in range(1, 7)), encoding="utf-8")
        pipeline = {"seed": 42, "data": {"raw_path": "data/raw/steam.csv",
                    "processed_dir": "data/processed", "min_games_per_user": 5},
                    "baselines": {"artifacts_dir": "data/baselines"}}
        (self.root / "config/default.yaml").write_text(yaml.safe_dump(pipeline))
        run_pipeline(PipelineConfig(raw, self.root / "data/processed", seed=42))
        from src.verbalize import read_rows
        mapping = json.loads((self.root / "data/processed/game_mapping.json").read_text())
        train = read_rows(self.root / "data/processed/train.csv", mapping)
        fit, validation = leave_one_out(train, 43)
        baseline = self.root / "data/baselines"
        baseline.mkdir()
        for name, rows in (("validation_fit.csv", fit), ("validation.csv", validation)):
            write_csv(baseline / name, ["user_id", "game_id", "game_title", "hours"],
                      ((row.user_id, mapping[row.game_title], row.game_title, row.hours) for row in rows))
        write_json(baseline / "user_mapping.json", {"1": 0, "2": 1, "3": 2})
        from src.data import ARTIFACT_NAMES
        write_json(baseline / "manifest.json", {
            "identity": {"input_sha256": {name: file_hash(self.root / "data/processed" / name) for name in ARTIFACT_NAMES}},
            "artifact_sha256": {name: file_hash(baseline / name) for name in
                                ("validation_fit.csv", "validation.csv", "user_mapping.json")},
        })
        tokenizer_dir = self.root / "data/tokenizer"
        tokenizer_dir.mkdir()
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "<bos>": 1, "Game": 2, "hrs": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # Even a tokenizer asset with truncation enabled must not conceal long histories.
        tokenizer.enable_truncation(2)
        tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
        write_json(tokenizer_dir / "tokenizer_config.json", {"bos_token": "<bos>"})
        self.document = {"pipeline_config": "config/default.yaml", "artifacts_dir": "data/verbalized",
                         "results_dir": "results/verbalization", "strategy": "top_hours", "k": 15,
                         "max_tokens": 20, "review_samples": 10,
                         "tokenizer": {"repository": "test/test", "revision": "a" * 40,
                                       "directory": "data/tokenizer", "sha256": {
                                           name: file_hash(tokenizer_dir / name) for name in
                                           ("tokenizer.json", "tokenizer_config.json")}}}
        self.path = self.root / "config/verbalize.yaml"
        self.path.write_text(yaml.safe_dump(self.document))
        self.config = load_config(self.path)

    def test_explicit_single_bos_and_no_silent_truncation(self):
        tokenizer = GemmaTokenizer(self.config["tokenizer_dir"])
        self.assertEqual(tokenizer.encode("Game hrs Game hrs"), [1, 2, 3, 2, 3])
        self.assertEqual(tokenizer.measure("Game hrs Game hrs"), {"text_tokens": 4, "input_tokens": 5})

    def test_split_sources_records_and_budget_flags(self):
        sources = load_sources(self.config)
        tokenizer = GemmaTokenizer(self.config["tokenizer_dir"])
        for split, expected_size in (("test", 5), ("validation", 4)):
            records = build_records(sources, tokenizer, split=split, k=15, strategy="top_hours", max_tokens=1)
            self.assertEqual(len(records), 3)
            for row in records:
                self.assertFalse(set(row["game_ids"]) & set(row["excluded_game_ids"]))
                self.assertEqual(row["rendered_games"], expected_size)
                self.assertEqual(row["history_games"], expected_size)
                self.assertEqual(row["excluded_from_source"], 0)
                self.assertTrue(row["over_budget"])
                self.assertEqual(row["input_tokens"], row["text_tokens"] + 1)
            self.assertEqual(summarize_records(records)["over_budget"], 3)

    def test_complete_run_cache_rejection_and_fresh_run_reproducibility(self):
        first = run_verbalizer(self.config)
        self.assertEqual(first, run_verbalizer(self.config))
        original = self.config["artifacts_dir"]
        other_config = {**self.config, "artifacts_dir": original.with_name("second-run")}
        self.assertEqual(first, run_verbalizer(other_config))
        for path in original.iterdir():
            self.assertEqual(path.read_bytes(), (other_config["artifacts_dir"] / path.name).read_bytes())
        identity = make_identity(self.config, load_sources(self.config))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            verify_artifacts(original, {**identity, "seed": 100})
        (original / "test_histories.jsonl").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Modified"):
            verify_artifacts(original, identity)

    def test_modified_input_or_membership_is_rejected(self):
        path = self.config["baseline_dir"] / "validation.csv"
        path.write_text(path.read_text() + "1,0,game 1,1\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            load_sources(self.config)

    def test_missing_tokenizer_stays_offline_and_modified_tokenizer_fails(self):
        tokenizer = self.config["tokenizer_dir"] / "tokenizer.json"
        tokenizer.write_text("modified")
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_tokenizer(self.config)
        tokenizer.unlink()
        with patch("src.verbalize.urllib.request.urlopen") as network:
            with self.assertRaisesRegex(FileNotFoundError, "download-tokenizer"):
                prepare_tokenizer(self.config)
            network.assert_not_called()

    def test_unknown_keys_and_unsafe_paths_fail(self):
        for change in ({"extra": 1}, {"k": 0}, {"strategy": "unknown"},
                       {"artifacts_dir": "data/processed"}, {"results_dir": "data/baselines"},
                       {"tokenizer": {**self.document["tokenizer"], "revision": "main"}}):
            self.path.write_text(yaml.safe_dump({**self.document, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_config(self.path)


if __name__ == "__main__":
    unittest.main()
