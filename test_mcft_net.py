import unittest
import numpy as np
import torch
from mcft_net import MCFTNet, make_input, tiled


class MCFTNetTests(unittest.TestCase):
    def test_zero_initialized_network_is_bicubic_identity(self):
        model=MCFTNet(width=8,blocks=1).eval();x=torch.rand(1,8,32,48);base=x[:,:3]
        with torch.inference_mode():result=model(x,base)
        torch.testing.assert_close(result,base)

    def test_feature_layout_and_anchor_distance(self):
        base=np.zeros((16,24,3),np.uint8);candidate=np.full_like(base,10)
        feature=make_input(base,candidate,5,[0,10])
        self.assertEqual(feature.shape,(16,24,8))
        self.assertTrue(np.allclose(feature[...,7],0.25))

    def test_full_frame_inference_path(self):
        model=MCFTNet(width=8,blocks=1).eval()
        feature=np.zeros((16,24,8),np.float32);feature[...,:3]=0.25
        result=tiled(model,feature,torch.device('cpu'),tile=0,overlap=0)
        self.assertEqual(result.shape,(16,24,3))
        self.assertTrue(np.all(result==63))


if __name__=='__main__':unittest.main()
