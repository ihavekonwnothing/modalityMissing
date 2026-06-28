import unittest

import torch

from train import _apply_controlled_optical_missing


class OpticalMissingScheduleTest(unittest.TestCase):
    def _batch(self):
        opt = torch.ones(2, 4, 8, 8)
        sar = torch.full((2, 2, 8, 8), 3.0)
        return {
            "image": torch.cat([sar, opt], dim=1),
            "sar": sar.clone(),
            "opt": opt.clone(),
            "mask": torch.zeros(2, 1, 8, 8),
            "valid_mask": torch.ones(2, 1, 8, 8),
        }

    def test_full_missing_sets_optical_to_zero_and_availability_to_zero(self):
        cfg = {"robust_fusion": {"enabled": True, "p_clean": 0.0, "p_optical_partial_missing": 0.0, "p_optical_full_missing": 1.0}}

        out = _apply_controlled_optical_missing(self._batch(), cfg)

        self.assertTrue(torch.equal(out["opt"], torch.zeros_like(out["opt"])))
        self.assertTrue(torch.equal(out["image"][:, 2:], torch.zeros_like(out["image"][:, 2:])))
        self.assertTrue(torch.equal(out["opt_mask"], torch.zeros(2, 1, 8, 8)))
        self.assertTrue(torch.equal(out["sar"], torch.full((2, 2, 8, 8), 3.0)))

    def test_clean_keeps_optical_and_availability_is_one(self):
        cfg = {"robust_fusion": {"enabled": True, "p_clean": 1.0, "p_optical_partial_missing": 0.0, "p_optical_full_missing": 0.0}}

        out = _apply_controlled_optical_missing(self._batch(), cfg)

        self.assertTrue(torch.equal(out["opt"], torch.ones_like(out["opt"])))
        self.assertTrue(torch.equal(out["image"][:, 2:], torch.ones_like(out["image"][:, 2:])))
        self.assertTrue(torch.equal(out["opt_mask"], torch.ones(2, 1, 8, 8)))

    def test_partial_missing_uses_availability_mask(self):
        cfg = {"robust_fusion": {"enabled": True, "p_clean": 0.0, "p_optical_partial_missing": 1.0, "p_optical_full_missing": 0.0, "mask_ratios": [0.5], "mask_types": ["random_block"]}}
        torch.manual_seed(4)

        out = _apply_controlled_optical_missing(self._batch(), cfg)

        self.assertEqual(float(out["opt_mask"].min()), 0.0)
        self.assertEqual(float(out["opt_mask"].max()), 1.0)
        self.assertTrue(torch.equal(out["opt"], out["image"][:, 2:]))


if __name__ == "__main__":
    unittest.main()
