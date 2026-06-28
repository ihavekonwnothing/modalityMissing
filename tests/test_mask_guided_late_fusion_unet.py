import unittest

import torch

from models.factory import build_model, select_model_input
from models.mask_guided_late_fusion_unet import MaskGuidedFusionBlock, MaskGuidedLateFusionUNet


class MaskGuidedLateFusionUNetTest(unittest.TestCase):
    def test_fusion_block_falls_back_to_sar_when_optical_is_missing(self):
        block = MaskGuidedFusionBlock(channels=8)
        sar = torch.randn(2, 8, 16, 16)
        opt = torch.randn(2, 8, 16, 16)
        opt_mask = torch.zeros(2, 1, 64, 64)

        fused = block(sar, opt, opt_mask)

        self.assertTrue(torch.allclose(fused, sar, atol=1e-6))

    def test_model_forward_accepts_modalities_and_mask(self):
        model = MaskGuidedLateFusionUNet(base_channels=8)
        sar = torch.randn(2, 2, 64, 64)
        opt = torch.randn(2, 4, 64, 64)
        opt_mask = torch.ones(2, 1, 64, 64)

        logits = model(sar, opt, opt_mask)

        self.assertEqual(tuple(logits.shape), (2, 1, 64, 64))

    def test_factory_builds_model_and_selects_mask_guided_input(self):
        model = build_model("mask_guided_late_fusion_unet", {"model_base_channels": 8})
        batch = {
            "image": torch.zeros(1, 6, 32, 32),
            "sar": torch.ones(1, 2, 32, 32),
            "opt": torch.full((1, 4, 32, 32), 2.0),
            "opt_mask": torch.zeros(1, 1, 32, 32),
        }

        selected = select_model_input(batch, "mask_guided_late_fusion_unet")

        self.assertIsInstance(model, MaskGuidedLateFusionUNet)
        self.assertEqual(len(selected), 3)
        self.assertTrue(torch.equal(selected[0], batch["sar"]))
        self.assertTrue(torch.equal(selected[1], batch["opt"]))
        self.assertTrue(torch.equal(selected[2], batch["opt_mask"]))


if __name__ == "__main__":
    unittest.main()
