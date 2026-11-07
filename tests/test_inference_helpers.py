import numpy as np
import unittest
import torch
import cv2
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from mcft_net import blend_prediction, fullframe_batch, build_model, infer, observation_downsample, tiled


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

    def test_tile_zero_preserves_float_until_blend(self):
        model = build_model('confidence', 4, 1).eval()
        feature = np.zeros((16, 16, 11), dtype=np.float32)
        result = tiled(model, feature, torch.device('cpu'), 0, 0)
        self.assertEqual(result.dtype, np.float32)

    def test_end_to_end_video_batch_sizes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = [np.full((32, 32, 3), 40 + i * 20, np.uint8) for i in range(3)]
            def write(path, values, size):
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 25, size)
                self.assertTrue(writer.isOpened())
                for frame in values: writer.write(cv2.resize(frame, size))
                writer.release()
            candidate, lr = root/'candidate.mkv', root/'lr.mkv'
            write(candidate, frames, (32, 32)); write(lr, frames, (16, 16))
            anchors = root/'anchors.json'; anchors.write_text(json.dumps({'anchor_indices': []}))
            model_path = root/'model.pth'
            torch.save({'state': build_model('confidence', 4, 1).state_dict(), 'width': 4,
                        'blocks': 1, 'architecture': 'confidence', 'degradation': 'bicubic'}, model_path)
            outputs = []
            for batch_size in (1, 4):
                output = root/f'out{batch_size}.mkv'
                infer(SimpleNamespace(model=str(model_path), candidate=str(candidate),
                    low_resolution=str(lr), anchors=str(anchors), output=str(output),
                    base_video=None, anchor_mode='true', degradation='bicubic', tile=0,
                    overlap=0, blend_alpha=0.61, device='cpu', max_frames=0,
                    infer_batch_size=batch_size))
                cap = cv2.VideoCapture(str(output)); read=[]
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    read.append(frame)
                cap.release(); outputs.append(read)
            self.assertEqual(len(outputs[0]), 3); self.assertEqual(len(outputs[1]), 3)
            for left, right in zip(outputs[0], outputs[1]): np.testing.assert_array_equal(left, right)

    def test_bicubic_observation_operator_has_expected_shape(self):
        prediction = torch.rand(2, 3, 20, 24)
        result = observation_downsample(prediction, 'bicubic', (10, 12))
        self.assertEqual(tuple(result.shape), (2, 3, 10, 12))
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_bicubic_operator_agrees_with_opencv_within_tolerance(self):
        rng = np.random.default_rng(11)
        image = rng.random((23, 31, 3), dtype=np.float32)
        cv_result = cv2.resize(image, (13, 10), interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
        torch_result = observation_downsample(tensor, 'bicubic', (10, 13))[0]
        torch_result = torch_result.permute(1, 2, 0).numpy()
        delta = np.abs(cv_result - torch_result)
        self.assertLess(float(delta.mean()), 0.01)
        self.assertLess(float(delta.max()), 0.08)


if __name__ == '__main__':
    unittest.main(verbosity=2)
