from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from torch.utils.data import DataLoader

from datasets.s1s2_water import S1S2WaterDataset, resolve_s1s2_root
from datasets.s1s2_water_cache import S1S2WaterPatchCacheDataset, is_patch_cache_ready
from evaluate import resolve_model_name
from models.factory import build_model, select_model_input
from models.output_utils import model_inference_mode, resolve_segmentation_logits
from train import _torch_optical_missing
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config
from utils.metrics import BinaryConfusion
from utils.test_records import append_test_records
import torch
from tqdm.auto import tqdm


def _stable_seed(sample_id: str, metadata: dict, mask_type: str, mask_ratio: float, base_seed: int) -> int:
    tile = metadata.get("tile", {})
    key = f"{base_seed}|{sample_id}|{tile.get('x', 0)}|{tile.get('y', 0)}|{mask_type}|{mask_ratio:.4f}"
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def _degrade_batch(batch, mask_type: str, mask_ratio: float, base_seed: int):
    opt = batch["opt"]
    out = dict(batch)
    out["opt"] = opt.clone()
    out["image"] = batch["image"].clone()
    out["opt_mask"] = torch.ones((opt.shape[0], 1, opt.shape[-2], opt.shape[-1]), device=opt.device, dtype=opt.dtype)
    for i in range(opt.shape[0]):
        seed = _stable_seed(batch["sample_id"][i], batch["metadata"][i], mask_type, mask_ratio, base_seed)
        torch.manual_seed(seed)
        out["opt"][i], out["opt_mask"][i, 0] = _torch_optical_missing(out["opt"][i], mask_type, mask_ratio)
    out["image"][:, 2:] = out["opt"]
    return out


def _build_dataset(config, split: str, stats, cache_dir: str | None, exclude_scenes: list[str]):
    if cache_dir and is_patch_cache_ready(cache_dir):
        return S1S2WaterPatchCacheDataset(cache_dir, split, exclude_scenes=exclude_scenes)
    root = resolve_s1s2_root(root_env=config["dataset"].get("root_env", "S1S2_WATER_ROOT"))
    return S1S2WaterDataset(
        root,
        split,
        patch_size=int(config["dataset"].get("patch_size", 256)),
        stats=stats,
        training=False,
        stride=config["dataset"].get("eval_stride"),
        exclude_scenes=exclude_scenes,
    )


def _conditions_from_args(args) -> list[tuple[str, float, str]]:
    if args.suite:
        return [
            ("clean", 0.0, "clean"),
            ("random_block_mask", 0.25, "random_block_25"),
            ("random_block_mask", 0.50, "random_block_50"),
            ("random_block_mask", 0.75, "random_block_75"),
            ("cloud_like_mask", 0.25, "cloud_like_25"),
            ("cloud_like_mask", 0.50, "cloud_like_50"),
            ("cloud_like_mask", 0.75, "cloud_like_75"),
            ("full_optical_missing", 1.0, "full_missing_100"),
        ]
    return [(args.mask_type, ratio, f"{args.mask_type}_{ratio:g}") for ratio in args.mask_ratios]


@torch.no_grad()
def evaluate_degradation(
    config,
    model_name: str | None,
    checkpoint: str,
    conditions: list[tuple[str, float, str]],
    split: str,
    cache_dir: str | None,
    exclude_scenes: list[str],
    batch_size: int | None,
    num_workers: int | None,
    max_batches: int | None,
    seed: int,
    progress: bool = True,
):
    output_dir = ensure_dir(config.get("output_dir", "outputs/s1s2_water/6band_main"))
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_name = resolve_model_name(config, model_name, ckpt)
    stats = ckpt.get("stats") or json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    ds = _build_dataset(config, split, stats, cache_dir, exclude_scenes)
    loader = DataLoader(
        ds,
        batch_size=batch_size or int(config["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else config["training"].get("num_workers", 0)),
        collate_fn=segmentation_collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meters = {condition: BinaryConfusion() for _, _, condition in conditions}
    run_conditions = [conditions[0]] if model_name == "s1_only_unet" else conditions
    total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    iterator = tqdm(loader, total=total_batches, desc=f"{split} degradation", dynamic_ncols=True, disable=not progress)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        for mask_type, mask_ratio, condition in run_conditions:
            degraded_batch = _degrade_batch(batch, mask_type, mask_ratio, seed)
            output = model(select_model_input(degraded_batch, model_name))
            logits = resolve_segmentation_logits(output, degraded_batch, model_inference_mode(config))
            meters[condition].update(logits, degraded_batch["mask"], degraded_batch["valid_mask"])
        if progress:
            iterator.set_postfix(condition=run_conditions[-1][2])
    if model_name == "s1_only_unet":
        base = meters[conditions[0][2]]
        for _, _, condition in conditions[1:]:
            meters[condition] = base
    rows = [
        {
            "model": model_name,
            "checkpoint": str(checkpoint),
            "split": split,
            "condition": condition,
            "mask_type": mask_type,
            "mask_ratio": mask_ratio,
                "excluded_scenes": ",".join(exclude_scenes),
                "evaluation": "degradation",
                "config": str(config.get("_config_path", "")),
                **meters[condition].compute(),
            }
        for mask_type, mask_ratio, condition in conditions
    ]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--mask_type", default="clean")
    parser.add_argument("--mask_ratios", nargs="+", type=float, default=[0.0])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--exclude-scene", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = args.config
    cache_dir = args.cache_dir or config.get("dataset", {}).get("cache_dir")
    exclude_scenes = [str(v) for v in (args.exclude_scene or config.get("dataset", {}).get("exclude_scenes", []))]
    conditions = _conditions_from_args(args)
    rows = evaluate_degradation(
        config=config,
        model_name=args.model,
        checkpoint=args.checkpoint,
        conditions=conditions,
        split=args.split,
        cache_dir=cache_dir,
        exclude_scenes=exclude_scenes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        seed=args.seed,
        progress=not args.no_progress,
    )

    output_dir = ensure_dir(config.get("output_dir", "outputs/s1s2_water/6band_main"))
    out_file = Path(args.output_csv) if args.output_csv else ensure_dir(output_dir / "metrics") / "degradation_suite_metrics.csv"
    ensure_dir(out_file.parent)
    write_header = (not args.append) or (not out_file.exists())
    mode = "a" if args.append else "w"
    fieldnames = ["model", "checkpoint", "split", "condition", "mask_type", "mask_ratio", "excluded_scenes", "IoU", "F1", "Precision", "Recall"]
    with out_file.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])
    append_test_records(rows)
    print(json.dumps({"output": str(out_file), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
