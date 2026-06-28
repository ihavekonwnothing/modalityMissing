import unittest

import torch

from tools.evaluate_fallback_ablation import fallback_logits


class FallbackRulesTest(unittest.TestCase):
    def test_global_scalar_fallback_uses_batch_availability(self):
        logits_fused = torch.full((2, 1, 2, 2), 10.0)
        logits_sar = torch.full((2, 1, 2, 2), 2.0)
        opt_mask = torch.zeros((2, 1, 2, 2))
        opt_mask[0].fill_(0.25)
        opt_mask[1].fill_(0.75)

        out = fallback_logits(logits_fused, logits_sar, opt_mask, "global_scalar")

        self.assertTrue(torch.allclose(out[0], torch.full((1, 2, 2), 4.0)))
        self.assertTrue(torch.allclose(out[1], torch.full((1, 2, 2), 8.0)))

    def test_pixelwise_fallback_uses_full_resolution_mask(self):
        logits_fused = torch.full((1, 1, 2, 2), 10.0)
        logits_sar = torch.full((1, 1, 2, 2), 2.0)
        opt_mask = torch.tensor([[[[1.0, 0.0], [0.5, 0.25]]]])

        out = fallback_logits(logits_fused, logits_sar, opt_mask, "pixelwise")

        expected = torch.tensor([[[[10.0, 2.0], [6.0, 4.0]]]])
        self.assertTrue(torch.allclose(out, expected))

    def test_pixelwise_fallback_resizes_mask_with_nearest_neighbor(self):
        logits_fused = torch.full((1, 1, 4, 4), 10.0)
        logits_sar = torch.full((1, 1, 4, 4), 2.0)
        opt_mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

        out = fallback_logits(logits_fused, logits_sar, opt_mask, "pixelwise")

        expected_mask = torch.tensor(
            [[
                [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
            ]]
        )
        expected = expected_mask * logits_fused + (1.0 - expected_mask) * logits_sar
        self.assertTrue(torch.allclose(out, expected))

    def test_clean_and_full_missing_are_identical_between_rules(self):
        logits_fused = torch.randn(2, 1, 8, 8)
        logits_sar = torch.randn(2, 1, 8, 8)
        clean = torch.ones(2, 1, 8, 8)
        missing = torch.zeros(2, 1, 8, 8)

        self.assertTrue(torch.equal(fallback_logits(logits_fused, logits_sar, clean, "global_scalar"), fallback_logits(logits_fused, logits_sar, clean, "pixelwise")))
        self.assertTrue(torch.equal(fallback_logits(logits_fused, logits_sar, missing, "global_scalar"), fallback_logits(logits_fused, logits_sar, missing, "pixelwise")))

    def test_backward_is_not_triggered_by_fallback(self):
        logits_fused = torch.randn(1, 1, 4, 4, requires_grad=True)
        logits_sar = torch.randn(1, 1, 4, 4, requires_grad=True)
        opt_mask = torch.ones(1, 1, 4, 4)

        out = fallback_logits(logits_fused, logits_sar, opt_mask, "pixelwise")

        self.assertTrue(out.requires_grad)
        self.assertIsNone(logits_fused.grad)
        self.assertIsNone(logits_sar.grad)


if __name__ == "__main__":
    unittest.main()
