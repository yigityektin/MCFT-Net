import unittest

import cv2
import numpy as np

from motion_frequency_transfer import (
    adjacent_anchors,
    anchor_high_frequency,
    frequency_bands,
    edge_orientation_agreement,
    scene_compatible,
    overlapping_patch_agreement,
    phase_preserving_spectral_transfer,
    super_resolution_residual,
    remap_with_displacement,
    resize_for_flow,
    resize_flow,
    compute_flow,
)
from mcft_constrained_reconstruct import content_adaptive_residual, gradients, local_affine_refine, multiframe_project, multiscale_residual, project_observation, project_observation_operator, screened_poisson
from mcft_temporal_bridge import adaptive_anchor_detail, bridge_luma


class MotionFrequencyTransferTests(unittest.TestCase):
    def test_backward_remap_direction(self):
        image = np.arange(20, dtype=np.float32).reshape(4, 5)
        displacement = np.zeros((4, 5, 2), np.float32)
        displacement[..., 0] = 1.0
        warped = remap_with_displacement(image, displacement, cv2.BORDER_CONSTANT)
        np.testing.assert_allclose(warped[:, :-1], image[:, 1:])

    def test_resize_flow_scales_vectors(self):
        flow = np.ones((2, 3, 2), np.float32)
        resized = resize_flow(flow, 6, 8)
        self.assertEqual(resized.shape, (8, 6, 2))
        np.testing.assert_allclose(resized[..., 0], 2.0)
        np.testing.assert_allclose(resized[..., 1], 4.0)

    def test_two_sided_anchor_lookup(self):
        self.assertEqual(adjacent_anchors(6, [0, 5, 10]), [5, 10])
        self.assertEqual(adjacent_anchors(0, [0, 5]), [0])

    def test_flow_resolution_scaling(self):
        image = np.zeros((48, 64, 3), np.uint8)
        self.assertEqual(resize_for_flow(image, 0.5).shape, (24, 32, 3))

    def test_dis_flow_shape_and_finiteness(self):
        a = np.zeros((32, 48, 3), np.uint8)
        cv2.circle(a, (20, 16), 5, (255, 255, 255), -1)
        b = np.roll(a, 1, axis=1)
        flow = compute_flow(a, b, "dis")
        self.assertEqual(flow.shape, (32, 48, 2))
        self.assertTrue(np.isfinite(flow).all())

    def test_sr_residual_does_not_duplicate_shared_edges(self):
        low = np.zeros((16, 16, 3), np.uint8)
        low[:, 8:] = 255
        sr = cv2.resize(low, (32, 32), interpolation=cv2.INTER_CUBIC)
        residual = super_resolution_residual(sr, low, 1.2)
        self.assertLess(abs(float(residual.mean())), 1e-4)
        self.assertLess(float(np.mean(np.abs(residual))), 0.5)

    def test_anchor_high_frequency_extracts_edges(self):
        image = np.zeros((32, 32, 3), np.uint8)
        image[:, 16:] = 255
        residual = anchor_high_frequency(image, 1.2)
        self.assertLess(abs(float(residual.mean())), 1e-4)
        self.assertGreater(float(np.abs(residual).max()), 1.0)

    def test_frequency_bands_reconstruct_luminance(self):
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        low, middle, high = frequency_bands(image, 0.8, 2.4)
        y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        np.testing.assert_allclose(low + middle + high, y, atol=1e-5)

    def test_edge_orientation_rejects_perpendicular_edges(self):
        vertical = np.zeros((32, 32), np.float32); vertical[:, 16:] = 255
        horizontal = np.zeros((32, 32), np.float32); horizontal[16:, :] = 255
        same = edge_orientation_agreement(vertical, vertical)
        cross = edge_orientation_agreement(vertical, horizontal)
        self.assertGreater(float(same[:, 15:17].mean()), float(cross[:, 15:17].mean()))

    def test_scene_compatibility(self):
        dark = np.zeros((32, 32, 3), np.uint8)
        light = np.full((32, 32, 3), 255, np.uint8)
        self.assertTrue(scene_compatible(dark, dark, 0.45))
        self.assertFalse(scene_compatible(dark, light, 0.45))

    def test_patch_agreement_prefers_matching_texture(self):
        rng = np.random.default_rng(2)
        texture = rng.normal(size=(32, 32)).astype(np.float32)
        unrelated = rng.normal(size=(32, 32)).astype(np.float32)
        same = overlapping_patch_agreement(texture, texture, 3)
        other = overlapping_patch_agreement(texture, unrelated, 3)
        self.assertGreater(float(same.mean()), float(other.mean()) + 0.5)

    def test_phase_preserving_identity_is_nearly_identity(self):
        rng = np.random.default_rng(4)
        target = rng.uniform(0, 255, (64, 64)).astype(np.float32)
        confidence = np.ones_like(target)
        result = phase_preserving_spectral_transfer(
            target, target, confidence, 32, 16, 0.9, 1.12, 0.25, 0.65
        )
        self.assertLess(float(np.mean(np.abs(result - target))), 1e-3)

    def test_screened_poisson_preserves_matching_gradient_field(self):
        yy, xx = np.mgrid[:32, :48]
        base = (2.0 * xx + 3.0 * yy).astype(np.float32)
        gx, gy = gradients(base)
        result = screened_poisson(base, gx, gy, 0.35, 35)
        np.testing.assert_allclose(result, base, atol=1e-4)

    def test_observation_projection_reduces_lr_residual(self):
        rng = np.random.default_rng(12)
        image = rng.uniform(0, 255, (64, 96)).astype(np.float32)
        observed = cv2.resize(image, (48, 32), interpolation=cv2.INTER_AREA)
        estimate = cv2.GaussianBlur(image, (0, 0), 2.0)
        before = np.mean(np.abs(observed-cv2.resize(estimate,(48,32),interpolation=cv2.INTER_AREA)))
        projected = project_observation(estimate, observed, 0.8)
        after = np.mean(np.abs(observed-cv2.resize(projected,(48,32),interpolation=cv2.INTER_AREA)))
        self.assertLess(after, before)

    def test_operator_projection_reduces_modeled_residual(self):
        rng=np.random.default_rng(31)
        truth=rng.uniform(0,255,(64,96)).astype(np.float32)
        model={'blur_sigma_hr':0.5,'phase_x_hr':0.0,'phase_y_hr':0.0}
        observed=cv2.resize(cv2.GaussianBlur(truth,(0,0),0.5),(48,32),interpolation=cv2.INTER_AREA)
        estimate=cv2.GaussianBlur(truth,(0,0),2.0)
        before=np.mean(np.abs(observed-cv2.resize(cv2.GaussianBlur(estimate,(0,0),0.5),(48,32),interpolation=cv2.INTER_AREA)))
        result=project_observation_operator(estimate,observed,0.8,model)
        after=np.mean(np.abs(observed-cv2.resize(cv2.GaussianBlur(result,(0,0),0.5),(48,32),interpolation=cv2.INTER_AREA)))
        self.assertLess(after,before)

    def test_affine_refinement_identity_is_stable(self):
        yy,xx=np.mgrid[:128,:128]
        image=(80+40*np.sin(xx/7)+30*np.cos(yy/9)).astype(np.float32)
        refined,confidence=local_affine_refine(image,image,np.ones_like(image),128,64)
        self.assertLess(float(np.mean(np.abs(refined-image))),0.05)
        self.assertGreater(float(confidence.mean()),0.9)

    def test_temporal_bridge_identity_anchor_is_stable(self):
        rng = np.random.default_rng(17)
        frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        observed = cv2.resize(frame, (48, 32), interpolation=cv2.INTER_AREA)
        bridged = bridge_luma(frame, observed, frame, 0.3, 0.8)
        self.assertLess(float(np.mean(np.abs(bridged.astype(float)-frame.astype(float)))), 0.6)

    def test_multiscale_residual_confidence_extremes(self):
        rng = np.random.default_rng(22)
        residual = rng.normal(size=(32, 48)).astype(np.float32)
        full = multiscale_residual(residual, np.ones_like(residual))
        np.testing.assert_allclose(full, residual, atol=1e-5)
        medium = multiscale_residual(residual, np.zeros_like(residual))
        self.assertLess(float(cv2.Laplacian(medium, cv2.CV_32F).var()),
                        float(cv2.Laplacian(residual, cv2.CV_32F).var()))

    def test_content_gate_suppresses_flat_region_fine_detail(self):
        rng=np.random.default_rng(52);residual=rng.normal(size=(32,48)).astype(np.float32)
        flat=np.full_like(residual,100);textured=rng.uniform(0,255,residual.shape).astype(np.float32)
        a=content_adaptive_residual(residual,np.ones_like(residual),flat)
        b=content_adaptive_residual(residual,np.ones_like(residual),textured)
        self.assertLess(float(cv2.Laplacian(a,cv2.CV_32F).var()),float(cv2.Laplacian(b,cv2.CV_32F).var()))

    def test_adaptive_anchor_detail_is_bounded(self):
        smooth = np.zeros((32, 48, 3), np.uint8)
        anchor = smooth.copy(); anchor[:, 24:] = 255
        self.assertEqual(adaptive_anchor_detail(None, anchor, smooth), 0.90)
        value = adaptive_anchor_detail(smooth, anchor, smooth)
        self.assertGreaterEqual(value, 0.85)
        self.assertLessEqual(value, 0.98)

    def test_multiframe_projection_identity_is_stable(self):
        yy, xx = np.mgrid[:64, :96]
        image = (60 + xx + 0.5 * yy).astype(np.float32)
        observed = cv2.resize(image, (48, 32), interpolation=cv2.INTER_AREA)
        result = multiframe_project(image, observed, [observed.copy()], 0.25)
        self.assertLess(float(np.mean(np.abs(result-image))), 1e-3)

    def test_multiframe_projection_rejects_photometric_outlier(self):
        image = np.full((64, 96), 64.0, np.float32)
        observed = np.full((32, 48), 64.0, np.float32)
        outlier = np.full((32, 48), 240.0, np.float32)
        result = multiframe_project(image, observed, [outlier], 0.25)
        self.assertLess(float(np.mean(np.abs(result-image))), 0.1)


if __name__ == "__main__":
    unittest.main()
