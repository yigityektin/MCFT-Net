import numpy as np
import unittest

from mcft_net import blend_prediction


class BlendTest(unittest.TestCase):
    def test_blend_is_conservative_and_deterministic(self):
        base = np.full((4, 5, 3), 100, dtype=np.uint8)
        prediction = np.full((4, 5, 3), 180, dtype=np.float32)
        single = blend_prediction(base, prediction, 0.5)
        batch = blend_prediction(base, prediction, 0.5)
        np.testing.assert_array_equal(single, batch)
        self.assertEqual(int(single[0, 0, 0]), 140)
