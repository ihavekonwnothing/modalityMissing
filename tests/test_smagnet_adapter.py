import unittest

import torch

from models.factory import build_model, select_model_input
from models.smagnet_adapter import SMAGNetAdapter


class SMAGNetAdapterTest(unittest.TestCase):
    def test_factory_builds_smagnet_and_selects_multimodal_inputs(self):
        model = build_model(
            "smagnet",
            {
                "model": {
                    "encoder_name": "resnet18",
                    "encoder_depth": 5,
                    "encoder_weights_sar": None,
                    "encoder_weights_msi": None,
                    "decoder_channels": [256, 128, 64, 32, 16],
                    "inference_mode": "adaptive_fallback",
                }
            },
        )
        batch = {
            "image": torch.zeros(1, 6, 64, 64),
            "sar": torch.randn(1, 2, 64, 64),
            "opt": torch.randn(1, 4, 64, 64),
            "opt_mask": torch.ones(1, 1, 64, 64),
        }

        selected = select_model_input(batch, "smagnet")

        self.assertIsInstance(model, SMAGNetAdapter)
        self.assertEqual(len(selected), 3)
        self.assertTrue(torch.equal(selected[0], batch["sar"]))
        self.assertTrue(torch.equal(selected[1], batch["opt"]))
        self.assertTrue(torch.equal(selected[2], batch["opt_mask"]))

    def test_smagnet_adapter_outputs_framework_dict_and_adaptive_fallback(self):
        model = build_model(
            "smagnet",
            {
                "model": {
                    "encoder_name": "resnet18",
                    "encoder_depth": 5,
                    "encoder_weights_sar": None,
                    "encoder_weights_msi": None,
                    "decoder_channels": [256, 128, 64, 32, 16],
                    "inference_mode": "adaptive_fallback",
                }
            },
        )
        sar = torch.randn(1, 2, 64, 64)
        opt = torch.randn(1, 4, 64, 64)

        missing = model(sar, opt, torch.zeros(1, 1, 64, 64))
        available = model(sar, opt, torch.ones(1, 1, 64, 64))

        self.assertEqual(tuple(missing["logits_fused"].shape), (1, 1, 64, 64))
        self.assertEqual(tuple(missing["logits_sar"].shape), (1, 1, 64, 64))
        self.assertEqual(tuple(missing["logits"].shape), (1, 1, 64, 64))
        self.assertTrue(torch.allclose(missing["logits"], missing["logits_sar"], atol=1e-6))
        self.assertTrue(torch.allclose(available["logits"], available["logits_fused"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
