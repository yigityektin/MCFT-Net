import unittest

import numpy as np
import torch

from mcft_hf_stream import degrade_x2, make_triplet_example
from mcft_net import build_model


class TestHFStreamPreprocessing(unittest.TestCase):
    def test_degradation_shapes(self):
        frame = np.zeros((65, 99, 3), np.uint8)
        lr, base = degrade_x2(frame)
        self.assertEqual(lr.shape, (32, 49, 3))
        self.assertEqual(base.shape, (64, 98, 3))

    def test_triplet_feature_is_compatible_with_confidence_net(self):
        yy, xx = np.mgrid[:64, :96]
        base = np.stack((xx % 256, yy % 256, (xx+yy) % 256), axis=2).astype(np.uint8)
        frames = [np.roll(base, shift, axis=1) for shift in (-1, 0, 1)]
        feature, target, lr, proposal = make_triplet_example(
            frames, flow_method="farneback", strength=0.5
        )
        self.assertEqual(feature.shape, (64, 96, 11))
        self.assertEqual(target.shape, proposal.shape)
        self.assertEqual(lr.shape, (32, 48, 3))
        self.assertTrue(np.isfinite(feature).all())
        self.assertGreaterEqual(float(feature.min()), 0.0)
        model = build_model("confidence", width=8, blocks=2).eval()
        tensor = torch.from_numpy(feature).permute(2, 0, 1)[None].float()
        with torch.inference_mode():
            prediction = model(tensor, tensor[:, :3])
        self.assertEqual(tuple(prediction.shape), (1, 3, 64, 96))
        self.assertTrue(bool(torch.isfinite(prediction).all()))


if __name__ == "__main__":
    unittest.main()
