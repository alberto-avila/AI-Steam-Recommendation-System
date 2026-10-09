"""CPU-only checks for training examples, negative sampling, masking, and the scorer adapter."""

import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.data import Interaction

if torch is not None:
    from src.train import CatalogScorer, Example, NegativeSampler, full_catalog_mask, popularity, training_examples


def sources():
    """Two users; user 1 holds out 'val' (validation) and 'test' (test)."""
    mapping = {"a": 0, "b": 1, "c": 2, "val": 3, "test": 4, "d": 5}
    display = {0: "A", 1: "B", 2: "C", 3: "VAL", 4: "TEST", 5: "D"}
    fit = [Interaction("1", "a", 50), Interaction("1", "b", 5), Interaction("1", "c", 1),
           Interaction("2", "a", 2), Interaction("2", "d", 3), Interaction("2", "b", 4)]
    validation = [Interaction("1", "val", 900), Interaction("2", "c", 1)]
    test = [Interaction("1", "test", 800), Interaction("2", "val", 7)]
    return {"mapping": mapping, "display": display, "fit": fit, "validation": validation,
            "test": test, "train": fit + validation, "users": ["1", "2"]}


@unittest.skipIf(torch is None, "torch not installed (pip install -e .[model])")
class TrainingExampleTests(unittest.TestCase):
    def setUp(self):
        self.sources = sources()
        self.examples = training_examples(self.sources, k=15, strategy="top_hours_summary")

    def test_one_example_per_fitting_interaction(self):
        self.assertEqual(len(self.examples), len(self.sources["fit"]))
        self.assertEqual([(ex.user_id, ex.target) for ex in self.examples],
                         [("1", 0), ("1", 1), ("1", 2), ("2", 0), ("2", 5), ("2", 1)])

    def test_target_and_held_out_games_never_enter_their_prompt(self):
        display = self.sources["display"]
        for ex in self.examples:
            self.assertNotIn(f"{display[ex.target]} (", ex.prompt)
            self.assertNotIn("VAL", ex.prompt)
            self.assertNotIn("TEST", ex.prompt)

    def test_summary_counts_exclude_target(self):
        # User 1 minus target 'a' leaves B (5) + C (1): 2 games, 6 hrs, top share 83%.
        self.assertEqual(self.examples[0].prompt,
                         "This player has played 2 games for 6 hrs in total.\n"
                         "Their most-played game accounts for 83% of their playtime.\n"
                         "Most-played games:\nB (5 hrs), C (1 hrs).")


@unittest.skipIf(torch is None, "torch not installed (pip install -e .[model])")
class SamplingAndMaskingTests(unittest.TestCase):
    def test_popularity_counts_fitting_users(self):
        counts = popularity(sources()["fit"], sources()["mapping"])
        self.assertEqual(counts.tolist(), [2, 2, 1, 0, 0, 1])

    def test_negatives_exclude_every_known_positive_and_never_repeat(self):
        counts = np.array([5., 5., 5., 5., 5., 5., 0.])
        sampler = NegativeSampler(counts, {"u": {0, 2, 4}}, exponent=0.75, seed=0)
        for _ in range(50):
            negatives = sampler.sample("u", 3).tolist()
            self.assertEqual(sorted(negatives), [1, 3, 5])  # Only valid pool; zero-popularity 6 excluded.

    def test_negative_sampling_is_seeded(self):
        counts = np.arange(1., 30.)
        draw = lambda: NegativeSampler(counts, {"u": {0}}, 0.75, seed=7).sample("u", 5).tolist()
        self.assertEqual(draw(), draw())

    def test_full_catalog_mask_keeps_target_and_hides_other_positives(self):
        eligible = torch.tensor([True, True, True, True, False])
        batch = [Example("u", 1, ""), Example("v", 0, "")]
        mask = full_catalog_mask(batch, {"u": {1, 2}, "v": {0}}, eligible)
        self.assertEqual(mask.tolist(), [[False, False, True, False, True],
                                         [False, False, False, False, True]])

    def test_catalog_scorer_indexes_requested_games_in_order(self):
        scorer = CatalogScorer(np.array([[0., 1., 2.], [5., 4., 3.]]), ["a", "b"])
        self.assertEqual(scorer.score("b", [2, 0]).tolist(), [3., 5.])


if __name__ == "__main__":
    unittest.main()
