import numpy as np

from mcft_net import blend_prediction


def test_blend_is_conservative_and_deterministic():
    base = np.full((4, 5, 3), 100, dtype=np.uint8)
    prediction = np.full((4, 5, 3), 180, dtype=np.float32)
    single = blend_prediction(base, prediction, 0.5)
    batch = blend_prediction(base, prediction, 0.5)
    np.testing.assert_array_equal(single, batch)
    assert int(single[0, 0, 0]) == 140

