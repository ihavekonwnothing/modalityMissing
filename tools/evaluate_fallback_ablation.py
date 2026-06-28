from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.s1s2_water_cache import S1S2WaterPatchCacheDataset
from datasets.sen1floods11_6band import Sen1Floods116BandDataset
from models.factory import build_model, select_model_input
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config
from utils.metrics import BinaryConfusion


FALLBACK_GLOBAL = "global_scalar"
FALLBACK_PIXELWISE = "pixelwise"
FALLBACKS = (FALLBACK_GLOBAL, FALLBACK_PIXELWISE)


@dataclass(frozen=True)
class Condition:
    condition: str
    mask_type: str
    mask_ratio: float


CONDITIONS = [
    Condition("clean", "clean", 0.0),
    Condition("block25", "random_block_mask", 0.25),
    Condition("block50", "random_block_mask", 0.50),
    Condition("block75", "random_block_mask", 0.75),
    Condition("cloud25", "cloud_like_mask", 0.25),
    Condition("cloud50", "cloud_like_mask", 0.50),
    Condition("cloud75", "cloud_like_mask", 0.75),
    Condition("full_missing", "full_optical_missing", 1.0),
]


def fallback_logits(logits_fused: torch.Tensor, logits_sar: torch.Tensor, opt_mask: torch.Tensor, mode: str) -> torch.Tensor:
    mask = opt_mask.to(device=logits_fused.device, dtype=logits_fused.dtype)
    if mask.shape[-2:] != logits_fused.shape[-2:]:
        mask = F.interpolate(mask, size=logits_fused.shape[-2:], mode="nearest")
    if mode == FALLBACK_GLOBAL:
        availability = mask.mean(dim=(1, 2, 3), keepdim=True)
        return availability * logits_fused + (1.0 - availability) * logits_sar
    if mode == FALLBACK_PIXELWISE:
        return mask * logits_fused + (1.0 - mask) * logits_sar
    raise ValueError(f"Unknown fallback mode: {mode}")


def _target_mask_pixels(height: int, width: int, ratio: float) -> int:
    return int(round(height * width * max(0.0, min(1.0, float(ratio)))))


def _random_block_missing_mask(height: int, width: int, ratio: float, device: torch.device) -> torch.Tensor:
    target = _target_mask_pixels(height, width, ratio)
    mask = torch.zeros((height, width), device=device, dtype=torch.bool)
    if target <= 0:
        return mask
    if target >= height * width:
        return torch.ones_like(mask)
    n_blocks = int(torch.randint(3, 13, (1,), device=device).item())
    attempts = 0
    while int(mask.sum().item()) < target and attempts < n_blocks * 20:
        attempts += 1
        remaining = max(1, target - int(mask.sum().item()))
        block_area = max(1, remaining // max(1, n_blocks))
        aspect = float((torch.rand(1, device=device) * 2.1 + 0.4).item())
        bh = int(max(8, min(height, round((block_area / aspect) ** 0.5))))
        bw = int(max(8, min(width, round(bh * aspect))))
        y0 = int(torch.randint(0, max(1, height - bh + 1), (1,), device=device).item())
        x0 = int(torch.randint(0, max(1, width - bw + 1), (1,), device=device).item())
        mask[y0 : y0 + bh, x0 : x0 + bw] = True
    current = int(mask.sum().item())
    if current > target:
        idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        keep = idx[torch.randperm(idx.numel(), device=device)[:target]]
        trimmed = torch.zeros_like(mask.flatten())
        trimmed[keep] = True
        mask = trimmed.view(height, width)
    return mask


def _cloud_like_missing_mask(height: int, width: int, ratio: float, device: torch.device) -> torch.Tensor:
    target = _target_mask_pixels(height, width, ratio)
    if target <= 0:
        return torch.zeros((height, width), device=device, dtype=torch.bool)
    if target >= height * width:
        return torch.ones((height, width), device=device, dtype=torch.bool)
    field = torch.randn((1, 1, height, width), device=device)
    kernel = max(7, (min(height, width) // 32) | 1)
    field = F.avg_pool2d(field, kernel_size=kernel, stride=1, padding=kernel // 2)
    kernel2 = max(3, kernel // 2) | 1
    field = F.avg_pool2d(field, kernel_size=kernel2, stride=1, padding=kernel2 // 2)
    flat = field.flatten()
    topk = torch.topk(flat, k=target, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[topk] = True
    return mask.view(height, width)


def _optical_missing(optical: torch.Tensor, mask_type: str, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = optical.shape[-2:]
    if ratio <= 0 or mask_type in {"clean", "none"}:
        missing = torch.zeros((height, width), device=optical.device, dtype=torch.bool)
    elif ratio >= 1 or mask_type == "full_optical_missing":
        missing = torch.ones((height, width), device=optical.device, dtype=torch.bool)
    elif mask_type == "random_block_mask":
        missing = _random_block_missing_mask(height, width, ratio, optical.device)
    elif mask_type == "cloud_like_mask":
        missing = _cloud_like_missing_mask(height, width, ratio, optical.device)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    out = optical.clone()
    out[:, missing] = 0.0
    return out, (~missing).to(dtype=optical.dtype)


def _stable_seed(key: str, condition: str, base_seed: int) -> int:
    digest = hashlib.sha1(f"{base_seed}|{key}|{condition}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _degrade_batch(batch: Dict[str, Any], condition: Condition, device: torch.device, base_seed: int) -> Dict[str, Any]:
    out = {k: v.to(device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
    opt = out["opt"].clone()
    opt_mask = torch.ones((opt.shape[0], 1, opt.shape[-2], opt.shape[-1]), device=device, dtype=opt.dtype)
    if condition.mask_ratio > 0:
        for i in range(opt.shape[0]):
            sample_key = str(out.get("chip_id", out.get("sample_id", [i]))[i])
            torch.manual_seed(_stable_seed(sample_key, condition.condition, base_seed))
            opt[i], opt_mask[i, 0] = _optical_missing(opt[i], condition.mask_type, condition.mask_ratio)
    out["opt"] = opt
    out["opt_mask"] = opt_mask
    out["image"] = torch.cat([out["sar"], opt], dim=1)
    if "label" in out and "mask" not in out:
        out["mask"] = out["label"]
    return out


def _apply_sar_preprocess(sar: torch.Tensor, stats: Dict[str, Any], mode: str) -> torch.Tensor:
    if mode in {"source_stats", "none"}:
        return sar
    source_mean = torch.as_tensor(stats["mean"][:2], device=sar.device, dtype=sar.dtype)[None, :, None, None]
    source_std = torch.as_tensor(stats["std"][:2], device=sar.device, dtype=sar.dtype)[None, :, None, None].clamp_min(1e-6)
    raw_sar = sar * source_std + source_mean
    if mode == "target_x100_source_stats":
        return (raw_sar * 100.0 - source_mean) / source_std
    if mode in {"target_image_zscore", "per_scene_zscore"}:
        mean = raw_sar.mean(dim=(-2, -1), keepdim=True)
        std = raw_sar.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (raw_sar - mean) / std
    if mode == "clip_-35_5":
        return (raw_sar.clamp(-35.0, 5.0) - source_mean) / source_std
    if mode == "clip_-30_0":
        return (raw_sar.clamp(-30.0, 0.0) - source_mean) / source_std
    raise ValueError(f"Unknown SAR preprocessing mode: {mode}")


def _collate_transfer(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = ["image", "sar", "opt", "label", "valid_mask"]
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in tensor_keys}
    batch["chip_id"] = [sample["chip_id"] for sample in samples]
    return batch


def _load_model(config: Dict[str, Any], checkpoint: str, device: torch.device) -> torch.nn.Module:
    model_name = str(config["model"]["name"])
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(model_name, config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _extract_branch_logits(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(output, dict) or "logits_fused" not in output or "logits_sar" not in output:
        raise TypeError("Fallback ablation requires model output dict with logits_fused and logits_sar.")
    return output["logits_fused"], output["logits_sar"]


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    conditions: Sequence[Condition],
    device: torch.device,
    seed: int,
    progress_desc: str,
) -> Dict[str, Dict[str, float]]:
    meters = {fallback: {condition.condition: BinaryConfusion() for condition in conditions} for fallback in FALLBACKS}
    total = len(loader)
    for batch in tqdm(loader, total=total, desc=progress_desc, dynamic_ncols=True):
        for condition in conditions:
            degraded = _degrade_batch(batch, condition, device, seed)
            output = model(select_model_input(degraded, model_name))
            logits_fused, logits_sar = _extract_branch_logits(output)
            for fallback in FALLBACKS:
                logits = fallback_logits(logits_fused, logits_sar, degraded["opt_mask"], fallback)
                meters[fallback][condition.condition].update(logits, degraded["mask"], degraded["valid_mask"])
    return {fallback: {condition: meters[fallback][condition].compute()["IoU"] for condition in meters[fallback]} for fallback in FALLBACKS}


def _write_wide_table(path: Path, results: Dict[str, Dict[str, float]]) -> None:
    ensure_dir(path.parent)
    fields = ["fallback", *[condition.condition for condition in CONDITIONS]]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for fallback in FALLBACKS:
            writer.writerow({"fallback": fallback, **{condition: f"{results[fallback][condition]:.8f}" for condition in fields[1:]}})


def _write_wide_markdown(path: Path, title: str, results: Dict[str, Dict[str, float]]) -> None:
    ensure_dir(path.parent)
    cols = [condition.condition for condition in CONDITIONS]
    lines = [f"# {title}\n\n", "| fallback | " + " | ".join(cols) + " |\n", "|---|" + "|".join(["---:"] * len(cols)) + "|\n"]
    for fallback in FALLBACKS:
        values = [f"{results[fallback][condition]:.4f}" for condition in cols]
        lines.append(f"| {fallback} | " + " | ".join(values) + " |\n")
    path.write_text("".join(lines), encoding="utf-8")


def _partial_delta_rows(dataset: str, results: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    rows = []
    for condition in [c.condition for c in CONDITIONS if c.condition not in {"clean", "full_missing"}]:
        global_iou = results[FALLBACK_GLOBAL][condition]
        pixel_iou = results[FALLBACK_PIXELWISE][condition]
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "global_scalar_iou": f"{global_iou:.8f}",
                "pixelwise_iou": f"{pixel_iou:.8f}",
                "delta_pixelwise_minus_global": f"{pixel_iou - global_iou:.8f}",
            }
        )
    return rows


def _assert_identity(dataset: str, results: Dict[str, Dict[str, float]], atol: float = 1e-12) -> None:
    for condition in ("clean", "full_missing"):
        diff = abs(results[FALLBACK_GLOBAL][condition] - results[FALLBACK_PIXELWISE][condition])
        if diff > atol:
            raise RuntimeError(
                f"{dataset} {condition} identity check failed: "
                f"global={results[FALLBACK_GLOBAL][condition]:.12f}, "
                f"pixelwise={results[FALLBACK_PIXELWISE][condition]:.12f}, diff={diff:.12g}"
            )


def _write_delta_outputs(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = out_dir / "fallback_partial_delta.csv"
    md_path = out_dir / "fallback_partial_delta.md"
    fields = ["dataset", "condition", "global_scalar_iou", "pixelwise_iou", "delta_pixelwise_minus_global"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Pixelwise vs Global Fallback Partial-Missing Delta\n\n", "| dataset | condition | global_scalar IoU | pixelwise IoU | delta |\n", "|---|---|---:|---:|---:|\n"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['condition']} | {float(row['global_scalar_iou']):.4f} | {float(row['pixelwise_iou']):.4f} | {float(row['delta_pixelwise_minus_global']):+.4f} |\n")
    md_path.write_text("".join(lines), encoding="utf-8")


def evaluate_s1s2(config: Dict[str, Any], model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> Dict[str, Dict[str, float]]:
    dataset_cfg = config.get("dataset", {})
    ds = S1S2WaterPatchCacheDataset(
        args.s1s2_cache_dir or dataset_cfg.get("cache_dir", "data/s1s2_water_patch_cache_512"),
        "test",
        exclude_scenes=dataset_cfg.get("exclude_scenes", []),
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size or config.get("training", {}).get("batch_size", 8)),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else config.get("training", {}).get("num_workers", 0)),
        collate_fn=segmentation_collate,
    )
    return evaluate_loader(model, str(config["model"]["name"]), loader, CONDITIONS, device, int(args.seed), "S1S2 fallback ablation")


def evaluate_transfer(config: Dict[str, Any], model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> Dict[str, Dict[str, float]]:
    stats = json.loads(Path(args.stats_path).read_text(encoding="utf-8"))
    ds = Sen1Floods116BandDataset(args.transfer_root, stats_path=args.stats_path)
    loader = DataLoader(
        ds,
        batch_size=int(args.transfer_batch_size or args.batch_size or 4),
        shuffle=False,
        num_workers=int(args.transfer_num_workers if args.transfer_num_workers is not None else args.num_workers if args.num_workers is not None else 4),
        collate_fn=_collate_transfer,
    )
    wrapped_batches = []
    for batch in loader:
        batch["sar"] = _apply_sar_preprocess(batch["sar"], stats, args.sar_preprocess)
        batch["mask"] = batch["label"]
        wrapped_batches.append(batch)
    return evaluate_loader(model, str(config["model"]["name"]), wrapped_batches, CONDITIONS, device, int(args.seed), "Sen1Floods11 fallback ablation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/s1s2_water/mask_aware_cross_attention_fusion.yaml")
    parser.add_argument("--checkpoint", default="outputs/s1s2_water/proposed/mask_aware_cross_attention_fusion_unet_ddp/checkpoints/best.ckpt")
    parser.add_argument("--s1s2-cache-dir", default=None)
    parser.add_argument("--transfer-root", default="transfer_dataset/Sen1Floods11_6band")
    parser.add_argument("--stats-path", default="data/s1s2_water_patch_cache_512/stats.json")
    parser.add_argument("--output-dir", default="outputs/pixelwise_ablation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--transfer-batch-size", type=int, default=None)
    parser.add_argument("--transfer-num-workers", type=int, default=None)
    parser.add_argument("--sar-preprocess", default="target_x100_source_stats", choices=["source_stats", "none", "target_x100_source_stats", "target_image_zscore", "per_scene_zscore", "clip_-35_5", "clip_-30_0"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is None:
        args.seed = int(config.get("training", {}).get("seed", 4))
    out_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(config, args.checkpoint, device)

    s1s2_results = evaluate_s1s2(config, model, args, device)
    _assert_identity("S1S2-Water", s1s2_results)
    _write_wide_table(out_dir / "s1s2_water_fallback_iou.csv", s1s2_results)
    _write_wide_markdown(out_dir / "s1s2_water_fallback_iou.md", "S1S2-Water Fallback IoU", s1s2_results)

    transfer_results = evaluate_transfer(config, model, args, device)
    _assert_identity("Sen1Floods11", transfer_results)
    _write_wide_table(out_dir / "sen1floods11_fallback_iou.csv", transfer_results)
    _write_wide_markdown(out_dir / "sen1floods11_fallback_iou.md", "Sen1Floods11 Fallback IoU", transfer_results)

    delta_rows = _partial_delta_rows("S1S2-Water", s1s2_results) + _partial_delta_rows("Sen1Floods11", transfer_results)
    _write_delta_outputs(out_dir, delta_rows)
    metadata = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "s1s2_cache_dir": args.s1s2_cache_dir or config.get("dataset", {}).get("cache_dir", "data/s1s2_water_patch_cache_512"),
        "transfer_root": args.transfer_root,
        "stats_path": args.stats_path,
        "sar_preprocess": args.sar_preprocess,
        "fallbacks": list(FALLBACKS),
        "conditions": [condition.__dict__ for condition in CONDITIONS],
        "no_training": True,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"s1s2": s1s2_results, "sen1floods11": transfer_results}, indent=2))


if __name__ == "__main__":
    main()
