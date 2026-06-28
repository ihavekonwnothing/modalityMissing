from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class BinaryConfusion:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def update(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> None:
        pred = (torch.sigmoid(logits) >= 0.5).float()
        valid = valid_mask > 0.5
        tgt = target > 0.5
        prd = pred > 0.5
        self.tp += float((prd & tgt & valid).sum().item())
        self.fp += float((prd & ~tgt & valid).sum().item())
        self.fn += float((~prd & tgt & valid).sum().item())

    def compute(self) -> Dict[str, float]:
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        return {"IoU": iou, "F1": f1, "Precision": precision, "Recall": recall}
