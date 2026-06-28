import unittest

import torch

from models.factory import build_model, select_model_input
from models.mask_aware_cross_attention_fusion_unet import (
    MaskAwareCrossAttentionBlock,
    MaskAwareCrossAttentionFusionUNet,
)
from models.output_utils import resolve_segmentation_logits


class MaskAwareCrossAttentionFusionUNetTest(unittest.TestCase):
    def test_cross_attention_block_preserves_shape(self):
        block = MaskAwareCrossAttentionBlock(16, hidden_dim=16, num_heads=4, window_size=4)
        sar = torch.randn(2, 16, 8, 8)
        opt = torch.randn(2, 16, 8, 8)
        opt_mask = torch.ones(2, 1, 64, 64)

        out = block(sar, opt, opt_mask)

        self.assertEqual(tuple(out.shape), tuple(sar.shape))

    def test_model_outputs_fused_and_sar_logits(self):
        model = MaskAwareCrossAttentionFusionUNet(base_channels=8, hidden_dim=16, num_heads=4, window_size=4)
        sar = torch.randn(2, 2, 64, 64)
        opt = torch.randn(2, 4, 64, 64)
        opt_mask = torch.ones(2, 1, 64, 64)

        out = model(sar, opt, opt_mask)

        self.assertEqual(tuple(out["logits_fused"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(out["logits_sar"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(out["logits"].shape), (2, 1, 64, 64))

    def test_model_adaptive_logits_match_sar_or_fused_at_extreme_masks(self):
        model = MaskAwareCrossAttentionFusionUNet(base_channels=8, hidden_dim=16, num_heads=4, window_size=4)
        sar = torch.randn(1, 2, 64, 64)
        opt = torch.randn(1, 4, 64, 64)

        missing = model(sar, opt, torch.zeros(1, 1, 64, 64))
        available = model(sar, opt, torch.ones(1, 1, 64, 64))

        self.assertTrue(torch.allclose(missing["logits"], missing["logits_sar"], atol=1e-6))
        self.assertTrue(torch.allclose(available["logits"], available["logits_fused"], atol=1e-6))

    def test_adaptive_fallback_uses_sar_when_optical_missing(self):
        logits_fused = torch.full((2, 1, 16, 16), 2.0)
        logits_sar = torch.full((2, 1, 16, 16), -3.0)
        batch = {"opt_mask": torch.zeros(2, 1, 16, 16)}

        logits = resolve_segmentation_logits({"logits_fused": logits_fused, "logits_sar": logits_sar}, batch, "adaptive_fallback")

        self.assertTrue(torch.equal(logits, logits_sar))

    def test_adaptive_fallback_uses_fused_when_optical_available(self):
        logits_fused = torch.full((2, 1, 16, 16), 2.0)
        logits_sar = torch.full((2, 1, 16, 16), -3.0)
        batch = {"opt_mask": torch.ones(2, 1, 16, 16)}

        logits = resolve_segmentation_logits({"logits_fused": logits_fused, "logits_sar": logits_sar}, batch, "adaptive_fallback")

        self.assertTrue(torch.equal(logits, logits_fused))

    def test_factory_builds_model_and_selects_inputs(self):
        model = build_model(
            "mask_aware_cross_attention_fusion_unet",
            {"model": {"hidden_dim": 16, "num_heads": 4, "window_size": 4}, "model_base_channels": 8},
        )
        batch = {
            "image": torch.zeros(1, 6, 32, 32),
            "sar": torch.ones(1, 2, 32, 32),
            "opt": torch.full((1, 4, 32, 32), 2.0),
            "opt_mask": torch.zeros(1, 1, 32, 32),
        }

        selected = select_model_input(batch, "mask_aware_cross_attention_fusion_unet")

        self.assertIsInstance(model, MaskAwareCrossAttentionFusionUNet)
        self.assertEqual(len(selected), 3)
        self.assertTrue(torch.equal(selected[0], batch["sar"]))
        self.assertTrue(torch.equal(selected[1], batch["opt"]))
        self.assertTrue(torch.equal(selected[2], batch["opt_mask"]))


if __name__ == "__main__":
    unittest.main()
