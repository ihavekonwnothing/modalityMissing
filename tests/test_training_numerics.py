import unittest

import torch

from train import (
    _clone_batchnorm_buffers,
    _model_state_has_nonfinite,
    _restore_batchnorm_buffers,
    _safe_segmentation_loss,
)


class TrainingNumericsTests(unittest.TestCase):
    def test_model_state_has_nonfinite_detects_batchnorm_buffers(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.BatchNorm2d(1))
        self.assertFalse(_model_state_has_nonfinite(model.state_dict()))
        model[1].running_mean[0] = float("nan")
        self.assertTrue(_model_state_has_nonfinite(model.state_dict()))

    def test_batchnorm_buffers_can_be_restored_after_bad_forward(self):
        bn = torch.nn.BatchNorm2d(2)
        model = torch.nn.Sequential(bn)
        snapshot = _clone_batchnorm_buffers(model)
        bn.running_mean.fill_(float("nan"))
        bn.running_var.fill_(float("nan"))
        _restore_batchnorm_buffers(model, snapshot)
        self.assertTrue(torch.isfinite(bn.running_mean).all())
        self.assertTrue(torch.isfinite(bn.running_var).all())

    def test_safe_segmentation_loss_is_float32_and_reports_nonfinite(self):
        logits = torch.tensor([[[[float("nan")]]]], dtype=torch.float16)
        target = torch.zeros((1, 1, 1, 1), dtype=torch.float16)
        valid = torch.ones((1, 1, 1, 1), dtype=torch.float16)
        loss, ok = _safe_segmentation_loss(logits, target, valid)
        self.assertFalse(ok)
        self.assertEqual(loss.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
