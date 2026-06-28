from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.s1s2_water import S1S2WaterDataset, estimate_train_stats, resolve_s1s2_root
from losses import segmentation_loss
from models.factory import build_model, select_model_input
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config
from utils.metrics import BinaryConfusion
from utils.seed import set_seed


def main():
    config = load_config("configs/s1s2_water/main_6band.yaml")
    set_seed(4)
    output_dir = ensure_dir(config.get("output_dir", "outputs/s1s2_water/6band_main"))
    metrics_dir = ensure_dir(output_dir / "metrics")
    root = resolve_s1s2_root(root_env=config["dataset"].get("root_env", "S1S2_WATER_ROOT"))
    stats = estimate_train_stats(root, patch_size=128, samples_per_scene=1, seed=4)
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    ds = S1S2WaterDataset(root, "train", patch_size=128, stats=stats, training=True, patches_per_scene=1, seed=4)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=segmentation_collate)
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model_name in config["models"]:
        model = build_model(model_name, {**config, "model_base_channels": 8}).to(device)
        batch_dev = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = model(select_model_input(batch_dev, model_name))
        loss = segmentation_loss(logits, batch_dev["mask"], batch_dev["valid_mask"])
        meter = BinaryConfusion()
        meter.update(logits, batch_dev["mask"], batch_dev["valid_mask"])
        rows.append({"model": model_name, "loss": float(loss.item()), **meter.compute()})
    out_file = metrics_dir / "quick_test_metrics.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "loss", "IoU", "F1", "Precision", "Recall"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"batch_image_shape": list(batch["image"].shape), "metrics": str(out_file)}, indent=2))


if __name__ == "__main__":
    main()
