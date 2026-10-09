"""CPU-only checks for pooling, vocabulary pruning, and the scoring head (no model weights)."""

import unittest
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # The model extra is optional; earlier checkpoints run without it.
    torch = None

if torch is not None:
    from src.model import GenRecModel, PrunedVocab, collate, parameter_counts, pool


class StubBackbone(nn.Module if torch else object):
    """Stands in for Gemma: hidden state = embedding of each token ID."""

    def __init__(self, vocab=10, hidden=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


@unittest.skipIf(torch is None, "torch not installed (pip install -e .[model])")
class PoolingTests(unittest.TestCase):
    def setUp(self):
        # Row 0 has 3 real tokens, row 1 has 1; padding positions hold large decoys.
        self.hidden = torch.tensor([[[1., 0.], [3., 0.], [5., 2.], [99., 99.]],
                                    [[2., 4.], [99., 99.], [99., 99.], [99., 99.]]])
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])

    def test_mean_pooling_ignores_padding(self):
        expected = torch.tensor([[3., 2 / 3], [2., 4.]])
        self.assertTrue(torch.allclose(pool(self.hidden, self.mask, "mean"), expected))

    def test_last_pooling_takes_final_real_token(self):
        expected = torch.tensor([[5., 2.], [2., 4.]])
        self.assertTrue(torch.equal(pool(self.hidden, self.mask, "last"), expected))

    def test_unknown_mode_fails(self):
        with self.assertRaises(ValueError):
            pool(self.hidden, self.mask, "max")

    def test_collate_right_pads_without_truncation(self):
        input_ids, mask = collate([[5, 6, 7], [8]], pad_id=0)
        self.assertEqual(input_ids.tolist(), [[5, 6, 7], [8, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 0, 0]])


@unittest.skipIf(torch is None, "torch not installed (pip install -e .[model])")
class PrunedVocabTests(unittest.TestCase):
    def test_remap_is_order_preserving_and_dense(self):
        vocab = PrunedVocab([900, 2, 0, 41, 2], original_size=1000)
        self.assertEqual(vocab.kept_ids, [0, 2, 41, 900])
        self.assertEqual(vocab.remap([2, 900, 0, 41, 41]), [1, 3, 0, 2, 2])

    def test_unknown_token_fails_loudly(self):
        vocab = PrunedVocab([0, 2], original_size=10)
        with self.assertRaisesRegex(KeyError, r"\[7\]"):
            vocab.remap([2, 7])


@unittest.skipIf(torch is None, "torch not installed (pip install -e .[model])")
class ScoringHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.backbone = StubBackbone()
        self.backbone.requires_grad_(False)  # Mirrors the frozen quantized base.
        self.model = GenRecModel(self.backbone, hidden_size=6, n_items=20, item_dim=4)
        self.input_ids, self.mask = collate([[1, 2, 3], [4, 5]], pad_id=0)

    def test_full_catalog_and_candidate_scores_agree(self):
        full = self.model(self.input_ids, self.mask)
        self.assertEqual(full.shape, (2, 20))
        candidates = torch.tensor([[3, 0, 19], [7, 7, 1]])
        sampled = self.model(self.input_ids, self.mask, candidates)
        self.assertEqual(sampled.shape, (2, 3))
        self.assertTrue(torch.allclose(sampled, full.gather(1, candidates)))

    def test_scores_ignore_padding_tokens(self):
        # Changing a padded position's token must not change that user's scores.
        changed = self.input_ids.clone()
        changed[1, 2] = 9
        self.assertTrue(torch.allclose(self.model(self.input_ids, self.mask),
                                       self.model(changed, self.mask)))

    def test_gradients_reach_head_but_not_frozen_backbone(self):
        loss = nn.functional.cross_entropy(self.model(self.input_ids, self.mask), torch.tensor([3, 5]))
        loss.backward()
        self.assertGreater(float(self.model.projection.weight.grad.norm()), 0)
        self.assertGreater(float(self.model.item_embedding.weight.grad.norm()), 0)
        self.assertIsNone(self.backbone.embed.weight.grad)
        counts = parameter_counts(self.model)
        self.assertEqual(counts["projection"], 6 * 4 + 4)
        self.assertEqual(counts["item_embedding"], 20 * 4)
        self.assertEqual(counts["frozen"], 10 * 6)

    def test_invalid_pooling_fails(self):
        with self.assertRaises(ValueError):
            GenRecModel(self.backbone, hidden_size=6, n_items=20, pooling="max")


if __name__ == "__main__":
    unittest.main()
