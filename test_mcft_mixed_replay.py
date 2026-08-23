import unittest
import torch
from mcft_mixed_replay import laplacian_overshoot,local_ssim_loss,texture_frequency_loss


class MixedReplayTests(unittest.TestCase):
    def test_edge_overshoot_zero_for_exact_target(self):
        value=torch.rand(2,3,16,16)
        self.assertAlmostEqual(float(laplacian_overshoot(value,value)),0.0,places=7)

    def test_edge_overshoot_detects_excess_detail(self):
        target=torch.zeros(1,3,16,16);prediction=target.clone();prediction[:,:,8,8]=1
        self.assertGreater(float(laplacian_overshoot(prediction,target)),0.0)

    def test_local_ssim_exact_and_degraded(self):
        target=torch.rand(2,3,16,16)
        self.assertAlmostEqual(float(local_ssim_loss(target,target)),0.0,places=5)
        blurred=torch.nn.functional.avg_pool2d(target,3,1,1)
        self.assertGreater(float(local_ssim_loss(blurred,target)),0.0)

    def test_texture_frequency_loss_exact_and_smoothed(self):
        target=torch.rand(2,3,16,16)
        self.assertAlmostEqual(float(texture_frequency_loss(target,target)),0.0,places=7)
        smoothed=torch.nn.functional.avg_pool2d(target,3,1,1)
        self.assertGreater(float(texture_frequency_loss(smoothed,target)),0.0)


if __name__=='__main__':unittest.main()
