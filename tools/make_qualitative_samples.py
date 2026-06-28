from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.s1s2_water import S1S2WaterDataset, resolve_s1s2_root
from datasets.transforms.optical_degradation import apply_optical_degradation
from models.factory import build_model, select_model_input
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config


def _norm_rgb(rgb: np.ndarray) -> np.ndarray:
    out = np.moveaxis(rgb, 0, -1)
    lo, hi = np.nanpercentile(out, [2, 98])
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s1s2_water/main_6band.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="proposed_robust_fusion_unet")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--mask-type", default="cloud_like_mask")
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(config.get("output_dir", "outputs/s1s2_water/6band_main"))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    root = resolve_s1s2_root(root_env=config["dataset"].get("root_env", "S1S2_WATER_ROOT"))
    ds = S1S2WaterDataset(root, "test", patch_size=int(config["dataset"].get("patch_size", 256)), stats=ckpt["stats"], training=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=segmentation_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    clean_dir = ensure_dir(output_dir / "figures" / "qualitative_clean")
    degraded_dir = ensure_dir(output_dir / "figures" / "qualitative_degraded")
    for idx, batch in enumerate(loader):
        if idx >= args.count:
            break
        batch_dev = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        clean_logits = model(select_model_input(batch_dev, args.model))
        clean_pred = (torch.sigmoid(clean_logits)[0, 0].cpu().numpy() >= 0.5).astype(float)
        opt_np = batch["opt"][0].numpy()
        degraded = apply_optical_degradation(opt_np, args.mask_type, args.mask_ratio, seed=idx)
        batch_dev["opt"] = torch.from_numpy(degraded.optical[None]).to(device)
        batch_dev["image"][:, 2:] = batch_dev["opt"]
        degraded_logits = model(select_model_input(batch_dev, args.model))
        degraded_pred = (torch.sigmoid(degraded_logits)[0, 0].cpu().numpy() >= 0.5).astype(float)
        sar = batch["sar"][0].numpy()
        mask = batch["mask"][0, 0].numpy()
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        axes[0, 0].imshow(sar[0], cmap="gray")
        axes[0, 0].set_title("SAR VV")
        axes[0, 1].imshow(_norm_rgb(opt_np[[2, 1, 0]]))
        axes[0, 1].set_title("RGB")
        axes[0, 2].imshow(mask, cmap="Blues")
        axes[0, 2].set_title("Ground truth")
        axes[1, 0].imshow(clean_pred, cmap="Blues")
        axes[1, 0].set_title("Clean prediction")
        axes[1, 1].imshow(degraded.missing_mask, cmap="gray")
        axes[1, 1].set_title("Degradation mask")
        axes[1, 2].imshow(degraded_pred, cmap="Blues")
        axes[1, 2].set_title("Degraded prediction")
        for ax in axes.ravel():
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(clean_dir / f"sample_{idx:03d}.png", dpi=150)
        fig.savefig(degraded_dir / f"sample_{idx:03d}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
