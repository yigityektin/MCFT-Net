import numpy as np
import unittest
import torch

from mcft_net import blend_prediction, fullframe_batch, build_model


class BlendTest(unittest.TestCase):
    def test_blend_is_conservative_and_deterministic(self):
        base = np.full((4, 5, 3), 100, dtype=np.uint8)
        prediction = np.full((4, 5, 3), 180, dtype=np.float32)
        single = blend_prediction(base, prediction, 0.5)
        batch = blend_prediction(base, prediction, 0.5)
        np.testing.assert_array_equal(single, batch)
        self.assertEqual(int(single[0, 0, 0]), 140)

    def test_fullframe_batch_size_one_matches_repeated_batch(self):
        torch.manual_seed(3)
        model = build_model('confidence', 4, 1).eval()
        feature = np.random.default_rng(3).random((16, 16, 11), dtype=np.float32)
        one = fullframe_batch(model, [feature], torch.device('cpu'))[0]
        four = fullframe_batch(model, [feature] * 4, torch.device('cpu'))
        np.testing.assert_allclose(four, np.repeat(one[None], 4, axis=0), rtol=1e-5, atol=1e-6)
