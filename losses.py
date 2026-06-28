from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_batch_pos_weight(target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.float()
    pos = (target.float() * valid).sum()
    neg = ((1.0 - target.float()) * valid).sum()
    return (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=100.0)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    if pos_weight is None:
        pos_weight = compute_batch_pos_weight(target, valid_mask).to(logits.device)
    elif not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none", pos_weight=pos_weight)
    valid = valid_mask.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    valid = valid_mask.float()
    target = target.float()
    intersection = (prob * target * valid).sum(dim=(1, 2, 3))
    denom = ((prob + target) * valid).sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    return masked_bce_with_logits(logits, target, valid_mask, pos_weight=pos_weight) + masked_dice_loss(logits, target, valid_mask)
