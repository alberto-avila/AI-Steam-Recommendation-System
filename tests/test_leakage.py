"""Split- and prompt-level holdout leakage checks."""

import unittest
from collections import Counter

from src.data import Interaction, leave_one_out
from src.verbalize import render_history


class SplitLeakageTests(unittest.TestCase):
    def test_test_and_validation_sentinels_never_appear_in_prompt(self):
        mapping = {"test secret": 0, "validation secret": 1, "known game": 2}
        display = {0: "TEST SECRET", 1: "VALIDATION SECRET", 2: "Known Game"}
        rows = [Interaction("1", "test secret", 10000),
                Interaction("1", "validation secret", 9000), Interaction("1", "known game", .1)]
        result = render_history(rows, mapping, display, excluded_game_ids={0, 1}, k=1)
        self.assertEqual(result.game_ids, (2,))
        self.assertNotIn("SECRET", result.text)
        self.assertNotIn("10000", result.text)
        self.assertNotIn("9000", result.text)
        self.assertEqual(result.text, "This player's most-played games:\nKnown Game (0.1 hrs).")

    def test_summary_statistics_ignore_held_out_games(self):
        # Counts, totals, and concentration must not reveal the held-out games either.
        mapping = {"test secret": 0, "validation secret": 1, "known game": 2, "other game": 3}
        display = {0: "TEST SECRET", 1: "VALIDATION SECRET", 2: "Known Game", 3: "Other Game"}
        known = [Interaction("1", "known game", .1), Interaction("1", "other game", .3)]
        render = lambda secret_hours: render_history(
            [Interaction("1", "test secret", secret_hours), Interaction("1", "validation secret", 9000), *known],
            mapping, display, excluded_game_ids={0, 1}, k=1, strategy="top_hours_summary").text
        self.assertEqual(render(10000), "This player has played 2 games for 0.4 hrs in total.\n"
                                        "Their most-played game accounts for 75% of their playtime.\n"
                                        "Most-played games (top 1 of 2):\nOther Game (0.3 hrs).")
        self.assertEqual(render(10000), render(1))

    def test_one_held_out_game_per_user_never_appears_in_training_history(self):
        interactions = [
            Interaction(str(user), f"game {game}", game + 0.5)
            for user in range(1, 25) for game in range(5 + user)
        ]
        train, test = leave_one_out(interactions, seed=42)
        train_pairs = {(row.user_id, row.game_title) for row in train}
        test_pairs = {(row.user_id, row.game_title) for row in test}
        all_pairs = {(row.user_id, row.game_title) for row in interactions}

        self.assertFalse(train_pairs & test_pairs)
        self.assertEqual(train_pairs | test_pairs, all_pairs)
        self.assertEqual(Counter(row.user_id for row in test), {str(user): 1 for user in range(1, 25)})
        self.assertTrue(all(count >= 4 for count in Counter(row.user_id for row in train).values()))
        self.assertCountEqual(train + test, interactions)


if __name__ == "__main__":
    unittest.main()
