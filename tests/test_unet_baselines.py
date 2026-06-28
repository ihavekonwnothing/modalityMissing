import unittest

import torch

from models.factory import build_model, select_model_input
from models.unet_baselines import S1OnlyUNet, S2OnlyUNet, UNetEfficientNetB0


class UNetBaselineTest(unittest.TestCase):
    def test_s1_and_s2_baselines_return_binary_logits(self):
        s1_model = S1OnlyUNet()
        s2_model = S2OnlyUNet()

        self.assertEqual(tuple(s1_model(torch.zeros(2, 2, 64, 64)).shape), (2, 1, 64, 64))
        self.assertEqual(tuple(s2_model(torch.zeros(2, 4, 64, 64)).shape), (2, 1, 64, 64))

    def test_factory_uses_configured_baseline_channels(self):
        s1 = build_model("s1_only_unet", {"model": {"encoder": "efficientnet-b0", "in_channels": 2}})
        s2 = build_model("s2_only_unet", {"model": {"encoder": "efficientnet-b0", "in_channels": 4}})

        self.assertIsInstance(s1, UNetEfficientNetB0)
        self.assertIsInstance(s2, UNetEfficientNetB0)

    def test_select_model_input_uses_single_modality_tensors(self):
        batch = {
            "image": torch.zeros(1, 6, 8, 8),
            "sar": torch.ones(1, 2, 8, 8),
            "opt": torch.full((1, 4, 8, 8), 2.0),
        }

        self.assertTrue(torch.equal(select_model_input(batch, "s1_only_unet"), batch["sar"]))
        self.assertTrue(torch.equal(select_model_input(batch, "s2_only_unet"), batch["opt"]))


if __name__ == "__main__":
    unittest.main()
