from __future__ import annotations

from typing import Any, Dict, List

import torch


def segmentation_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = ["image", "sar", "opt", "mask", "valid_mask"]
    batch: Dict[str, Any] = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in tensor_keys}
    batch["sample_id"] = [sample["sample_id"] for sample in samples]
    batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch
