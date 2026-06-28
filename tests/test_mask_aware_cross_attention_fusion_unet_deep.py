import unittest

import torch

from models.factory import build_model, select_model_input
from models.mask_aware_cross_attention_fusion_unet_deep import (
    DeepEncoder,
    MaskAwareCrossAttentionBlock,
    MaskAwareCrossAttentionFusionUNetDeep,
)


class MaskAwareCrossAttentionFusionUNetDeepTest(unittest.TestCase):
    def test_deep_encoder_produces_five_scales(self):
        encoder = DeepEncoder(2, base_channels=8)
        feats = encoder(torch.randn(2, 2, 64, 64))

        self.assertEqual(len(feats), 5)
        self.assertEqual(tuple(feats[0].shape), (2, 8, 64, 64))
        self.assertEqual(tuple(feats[1].shape), (2, 16, 32, 32))
        self.assertEqual(tuple(feats[2].shape), (2, 32, 16, 16))
        self.assertEqual(tuple(feats[3].shape), (2, 64, 8, 8))
        self.assertEqual(tuple(feats[4].shape), (2, 128, 4, 4))

    def test_cross_attention_block_preserves_shape(self):
        block = MaskAwareCrossAttentionBlock(64, hidden_dim=16, num_heads=4, window_size=4)
        sar = torch.randn(2, 64, 8, 8)
        opt = torch.randn(2, 64, 8, 8)
        opt_mask = torch.ones(2, 1, 64, 64)

        out = block(sar, opt, opt_mask)

        self.assertEqual(tuple(out.shape), tuple(sar.shape))

    def test_model_outputs_fused_and_sar_logits(self):
        model = MaskAwareCrossAttentionFusionUNetDeep(base_channels=8, hidden_dim=16, num_heads=4, window_size=4)
        sar = torch.randn(2, 2, 64, 64)
        opt = torch.randn(2, 4, 64, 64)
        opt_mask = torch.ones(2, 1, 64, 64)

        out = model(sar, opt, opt_mask)

        self.assertEqual(tuple(out["logits_fused"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(out["logits_sar"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(out["logits"].shape), (2, 1, 64, 64))

    def test_factory_builds_model_and_selects_inputs(self):
        model = build_model(
            "mask_aware_cross_attention_fusion_unet_deep",
            {"model": {"hidden_dim": 16, "num_heads": 4, "window_size": 4}, "model_base_channels": 8},
        )
        batch = {
            "image": torch.zeros(1, 6, 32, 32),
            "sar": torch.ones(1, 2, 32, 32),
            "opt": torch.full((1, 4, 32, 32), 2.0),
            "opt_mask": torch.zeros(1, 1, 32, 32),
        }

        selected = select_model_input(batch, "mask_aware_cross_attention_fusion_unet_deep")

        self.assertIsInstance(model, MaskAwareCrossAttentionFusionUNetDeep)
        self.assertEqual(len(selected), 3)
        self.assertTrue(torch.equal(selected[0], batch["sar"]))
        self.assertTrue(torch.equal(selected[1], batch["opt"]))
        self.assertTrue(torch.equal(selected[2], batch["opt_mask"]))


if __name__ == "__main__":
    unittest.main()
