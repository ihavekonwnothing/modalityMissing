import unittest

import torch

from models.factory import build_model, select_model_input
from models.mask_aware_cross_attention_fusion_unet import MaskAwareCrossAttentionNoSarAuxUNet
from models.mask_guided_late_fusion_unet import MaskGuidedLateFusionSarAuxUNet


class ComponentAblationModelsTest(unittest.TestCase):
    def test_cross_attention_no_sar_aux_outputs_fused_tensor(self):
        model = build_model(
            "mask_aware_cross_attention_no_sar_aux_unet",
            {"model": {"hidden_dim": 16, "num_heads": 4, "window_size": 4}, "model_base_channels": 8},
        )
        batch = {
            "image": torch.zeros(2, 6, 64, 64),
            "sar": torch.randn(2, 2, 64, 64),
            "opt": torch.randn(2, 4, 64, 64),
            "opt_mask": torch.ones(2, 1, 64, 64),
        }

        selected = select_model_input(batch, "mask_aware_cross_attention_no_sar_aux_unet")
        logits = model(selected)

        self.assertIsInstance(model, MaskAwareCrossAttentionNoSarAuxUNet)
        self.assertEqual(tuple(logits.shape), (2, 1, 64, 64))

    def test_mask_guided_sar_aux_adaptive_fallback_uses_sar_when_missing(self):
        model = build_model("mask_guided_late_fusion_sar_aux_unet", {"model_base_channels": 8})
        sar = torch.randn(1, 2, 64, 64)
        opt = torch.randn(1, 4, 64, 64)

        missing = model(sar, opt, torch.zeros(1, 1, 64, 64))
        available = model(sar, opt, torch.ones(1, 1, 64, 64))

        self.assertIsInstance(model, MaskGuidedLateFusionSarAuxUNet)
        self.assertEqual(tuple(missing["logits_fused"].shape), (1, 1, 64, 64))
        self.assertEqual(tuple(missing["logits_sar"].shape), (1, 1, 64, 64))
        self.assertTrue(torch.allclose(missing["logits"], missing["logits_sar"], atol=1e-6))
        self.assertTrue(torch.allclose(available["logits"], available["logits_fused"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
