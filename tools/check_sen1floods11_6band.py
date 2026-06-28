from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.sen1floods11_6band import Sen1Floods116BandDataset
from models.factory import build_model
from train import _torch_optical_missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check Sen1Floods11 6-band dataset and proposed model IO.")
    parser.add_argument("--data-root", default="transfer_dataset/Sen1Floods11_6band")
    parser.add_argument("--stats-path", default="data/s1s2_water_patch_cache_512/stats.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-samples", type=int, default=3)
    args = parser.parse_args()

    ds = Sen1Floods116BandDataset(args.data_root, stats_path=args.stats_path)
    if len(ds) != 446:
        raise AssertionError(f"Expected 446 samples, found {len(ds)}")

    checked = []
    for idx in range(min(args.max_samples, len(ds))):
        sample = ds[idx]
        assert tuple(sample["sar"].shape) == (2, 512, 512), sample["sar"].shape
        assert tuple(sample["opt"].shape) == (4, 512, 512), sample["opt"].shape
        assert tuple(sample["label"].shape) == (1, 512, 512), sample["label"].shape
        assert tuple(sample["valid_mask"].shape) == (1, 512, 512), sample["valid_mask"].shape
        checked.append(sample["chip_id"])

    sample = ds[0]
    opt_missing, opt_mask = _torch_optical_missing(sample["opt"], "full_optical_missing", 1.0)
    assert float(opt_mask.sum().item()) == 0.0
    assert float(opt_missing.abs().sum().item()) == 0.0

    model = build_model("mask_guided_late_fusion_unet", {})
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        logits = model(sample["sar"].unsqueeze(0), sample["opt"].unsqueeze(0), torch.ones_like(sample["label"]).unsqueeze(0))
    assert tuple(logits.shape) == (1, 1, 512, 512), logits.shape

    print(
        json.dumps(
            {
                "status": "ok",
                "samples": len(ds),
                "checked_chip_ids": checked,
                "forward_shape": list(logits.shape),
                "full_missing_opt_mask_sum": float(opt_mask.sum().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
