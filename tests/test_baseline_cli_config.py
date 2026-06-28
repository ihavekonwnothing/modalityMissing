import unittest

from train import resolve_model_name
from utils.config import load_config


class BaselineCliConfigTest(unittest.TestCase):
    def test_baseline_configs_define_expected_models(self):
        s1 = load_config("configs/s1s2_water/baseline_s1_unet_effb0.yaml")
        s2 = load_config("configs/s1s2_water/baseline_s2_unet_effb0.yaml")

        self.assertEqual(resolve_model_name(s1, None), "s1_only_unet")
        self.assertEqual(resolve_model_name(s2, None), "s2_only_unet")
        self.assertEqual(s1["model"]["encoder"], "efficientnet-b0")
        self.assertEqual(s2["model"]["encoder"], "efficientnet-b0")
        self.assertEqual(s1["model"]["input_bands"], ["VV", "VH"])
        self.assertEqual(s2["model"]["input_bands"], ["Blue", "Green", "Red", "NIR"])


if __name__ == "__main__":
    unittest.main()
