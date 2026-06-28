from __future__ import annotations

from typing import Any, Dict

import torch

from losses import segmentation_loss


def resolve_segmentation_logits(
    output: torch.Tensor | Dict[str, torch.Tensor],
    batch: Dict[str, Any] | None = None,
    inference_mode: str = "fused_only",
) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if not isinstance(output, dict):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if inference_mode == "adaptive_fallback":
        logits_fused = output["logits_fused"]
        logits_sar = output["logits_sar"]
        if batch is not None and batch.get("opt_mask") is not None:
            opt_mask = batch["opt_mask"].to(device=logits_fused.device, dtype=logits_fused.dtype)
            availability = opt_mask.mean(dim=(1, 2, 3), keepdim=True)
        elif output.get("opt_mask") is not None:
            opt_mask = output["opt_mask"].to(device=logits_fused.device, dtype=logits_fused.dtype)
            availability = opt_mask.mean(dim=(1, 2, 3), keepdim=True)
        else:
            availability = torch.ones((logits_fused.shape[0], 1, 1, 1), device=logits_fused.device, dtype=logits_fused.dtype)
        return availability * logits_fused + (1.0 - availability) * logits_sar
    if inference_mode == "sar_only":
        return output["logits_sar"]
    if inference_mode in {"fused_only", "fused"}:
        return output.get("logits", output["logits_fused"])
    raise ValueError(f"Unknown inference_mode: {inference_mode}")


def model_inference_mode(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("inference_mode", "fused_only"))
    return "fused_only"


def model_auxiliary_loss_weight(config: Dict[str, Any]) -> float:
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, dict):
        return float(model_cfg.get("lambda_sar", 0.0))
    return 0.0


def segmentation_output_loss(
    output: torch.Tensor | Dict[str, torch.Tensor],
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    config: Dict[str, Any],
) -> torch.Tensor:
    if torch.is_tensor(output):
        return segmentation_loss(output.float(), target.float(), valid_mask.float())
    if not isinstance(output, dict):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    fused = output.get("logits_fused", output.get("logits"))
    if fused is None:
        raise KeyError("Model output dict must contain logits_fused or logits")
    loss = segmentation_loss(fused.float(), target.float(), valid_mask.float())
    lambda_sar = model_auxiliary_loss_weight(config)
    if lambda_sar > 0 and output.get("logits_sar") is not None:
        loss = loss + lambda_sar * segmentation_loss(output["logits_sar"].float(), target.float(), valid_mask.float())
    return loss
