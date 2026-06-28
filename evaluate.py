from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.s1s2_water import S1S2WaterDataset, resolve_s1s2_root
from datasets.s1s2_water_cache import S1S2WaterPatchCacheDataset, is_patch_cache_ready
from models.factory import build_model, select_model_input
from models.output_utils import model_inference_mode, resolve_segmentation_logits
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config
from utils.metrics import BinaryConfusion
from utils.test_records import append_test_records


def resolve_model_name(config, model_name: str | None = None, checkpoint_data=None) -> str:
    if model_name:
        return model_name
    if checkpoint_data and checkpoint_data.get("model_name"):
        return str(checkpoint_data["model_name"])
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict) and model_cfg.get("name"):
        return str(model_cfg["name"])
    raise ValueError("Model name must be provided with --model, checkpoint model_name, or config.model.name")


@torch.no_grad()
def evaluate_model(
    config,
    model_name: str | None,
    checkpoint: str,
    split: str = "test",
    max_batches: int | None = None,
    output_csv: str | None = None,
):
    output_dir = ensure_dir(config.get("output_dir", "outputs/s1s2_water/6band_main"))
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_name = resolve_model_name(config, model_name, ckpt)
    stats = ckpt.get("stats") or json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    root = resolve_s1s2_root(root_env=config["dataset"].get("root_env", "S1S2_WATER_ROOT"))
    cache_dir = config["dataset"].get("cache_dir")
    if cache_dir and is_patch_cache_ready(cache_dir):
        ds = S1S2WaterPatchCacheDataset(cache_dir, split, exclude_scenes=config["dataset"].get("exclude_scenes", []))
    else:
        ds = S1S2WaterDataset(
            root,
            split,
            patch_size=int(config["dataset"].get("patch_size", 256)),
            stats=stats,
            training=False,
            exclude_scenes=config["dataset"].get("exclude_scenes", []),
        )
    loader = DataLoader(ds, batch_size=int(config["training"].get("batch_size", 16)), shuffle=False, num_workers=int(config["training"].get("num_workers", 0)), collate_fn=segmentation_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    overall = BinaryConfusion()
    per_scene = {}
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        sample_ids = batch["sample_id"]
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        output = model(select_model_input(batch, model_name))
        logits = resolve_segmentation_logits(output, batch, model_inference_mode(config))
        overall.update(logits, batch["mask"], batch["valid_mask"])
        for i, sid in enumerate(sample_ids):
            meter = per_scene.setdefault(sid, BinaryConfusion())
            meter.update(logits[i : i + 1], batch["mask"][i : i + 1], batch["valid_mask"][i : i + 1])
    metrics = overall.compute()
    metrics["model"] = model_name
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint)
    metrics["condition"] = "clean"
    metrics["mask_type"] = "clean"
    metrics["mask_ratio"] = 0.0
    metrics["excluded_scenes"] = ",".join(str(v) for v in config["dataset"].get("exclude_scenes", []))
    metrics["evaluation"] = "clean"
    metrics["config"] = str(config.get("_config_path", ""))
    metrics_dir = ensure_dir(output_dir / "metrics")
    clean_file = Path(output_csv) if output_csv else metrics_dir / "clean_test_metrics.csv"
    ensure_dir(clean_file.parent)
    write_header = not clean_file.exists()
    with clean_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "split", "IoU", "F1", "Precision", "Recall"])
        if write_header:
            writer.writeheader()
        writer.writerow({key: metrics[key] for key in ["model", "split", "IoU", "F1", "Precision", "Recall"]})
    per_scene_file = metrics_dir / "per_scene_metrics.csv"
    write_header = not per_scene_file.exists()
    with per_scene_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "split", "sample_id", "IoU", "F1", "Precision", "Recall"])
        if write_header:
            writer.writeheader()
        for sid, meter in per_scene.items():
            row = {"model": model_name, "split": split, "sample_id": sid, **meter.compute()}
            writer.writerow(row)
    append_test_records(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    config["_config_path"] = args.config
    metrics = evaluate_model(config, args.model, args.checkpoint, args.split, args.max_batches, args.output_csv)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
