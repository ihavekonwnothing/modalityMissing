from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.s1s2_water import S1S2WaterDataset, estimate_train_stats, resolve_s1s2_root
from losses import segmentation_loss
from models.factory import build_model, select_model_input
from train import resolve_model_name
from utils.collate import segmentation_collate
from utils.config import load_config
from utils.seed import set_seed


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=128)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = resolve_model_name(config, None)
    set_seed(int(config["training"].get("seed", 4)), bool(config["training"].get("deterministic", True)))
    root = resolve_s1s2_root(root_env=config["dataset"].get("root_env", "S1S2_WATER_ROOT"))
    stats = estimate_train_stats(root, patch_size=args.patch_size, samples_per_scene=1, seed=int(config["training"].get("seed", 4)))
    ds = S1S2WaterDataset(
        root,
        "train",
        patch_size=args.patch_size,
        stats=stats,
        training=True,
        patches_per_scene=1,
        train_sampling=config["dataset"].get("train_sampling", "random_crop"),
        stride=config["dataset"].get("train_stride"),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=segmentation_collate)
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, config).to(device)
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    x = select_model_input(batch, model_name)
    logits = model(x)
    loss = segmentation_loss(logits, batch["mask"], batch["valid_mask"])
    expected = (x.shape[0], int(config["model"].get("num_classes", 1)), x.shape[-2], x.shape[-1])
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Unexpected output shape {tuple(logits.shape)}, expected {expected}")
    print(
        json.dumps(
            {
                "model": model_name,
                "device": str(device),
                "input_shape": list(x.shape),
                "logits_shape": list(logits.shape),
                "loss": float(loss.item()),
                "uses_smp": bool(getattr(model, "uses_smp", False)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
