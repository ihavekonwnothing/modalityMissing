from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.sen1floods11_6band import Sen1Floods116BandDataset
from models.factory import build_model
from models.output_utils import model_inference_mode, resolve_segmentation_logits
from train import _torch_optical_missing
from utils.config import ensure_dir, load_config


@dataclass
class ConfusionWithTN:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update_from_tensors(self, pred: torch.Tensor, label: torch.Tensor, valid_mask: torch.Tensor) -> None:
        valid = valid_mask > 0.5
        pred_b = pred > 0.5
        label_b = label > 0.5
        self.tp += int((pred_b & label_b & valid).sum().item())
        self.fp += int((pred_b & ~label_b & valid).sum().item())
        self.fn += int((~pred_b & label_b & valid).sum().item())
        self.tn += int((~pred_b & ~label_b & valid).sum().item())

    def merge(self, other: "ConfusionWithTN") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    def compute(self) -> Dict[str, float | int]:
        iou = _safe_div(self.tp, self.tp + self.fp + self.fn)
        f1 = _safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = _safe_div(self.tp, self.tp + self.fn)
        return {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def _safe_div(num: int | float, denom: int | float) -> float:
    return float(num) / float(denom) if float(denom) > 0 else 0.0


def _collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = ["image", "sar", "opt", "label", "valid_mask"]
    batch = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in tensor_keys}
    for key in ("chip_id", "image_path", "label_path", "valid_mask_path"):
        if key in samples[0]:
            batch[key] = [sample[key] for sample in samples]
    return batch


def _stable_seed(chip_id: str, condition: str, base_seed: int) -> int:
    digest = hashlib.sha1(f"{base_seed}|{chip_id}|{condition}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _condition_name(mask_type: str, ratio: int) -> str:
    if ratio <= 0:
        return "clean"
    if ratio >= 100:
        return "full_100"
    return f"{mask_type}_{ratio}"


def _build_conditions(missing_ratios: Sequence[int], mask_types: Sequence[str]) -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    for ratio in missing_ratios:
        ratio = int(ratio)
        if ratio <= 0:
            if not any(c["condition"] == "clean" for c in conditions):
                conditions.append({"missing_ratio": 0, "mask_type": "clean", "condition": "clean"})
            continue
        if ratio >= 100:
            if not any(c["condition"] == "full_100" for c in conditions):
                conditions.append({"missing_ratio": 100, "mask_type": "full", "condition": "full_100"})
            continue
        for mask_type in mask_types:
            if mask_type == "full":
                continue
            if mask_type not in {"block", "cloud"}:
                raise ValueError(f"Unknown mask_type {mask_type!r}; expected block, cloud, or full")
            conditions.append({"missing_ratio": ratio, "mask_type": mask_type, "condition": _condition_name(mask_type, ratio)})
    return conditions


def _load_stats(stats_path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(stats_path).read_text(encoding="utf-8"))


def _apply_sar_preprocess(
    sar: torch.Tensor,
    sar_preprocess: str,
    stats: Dict[str, Any],
) -> torch.Tensor:
    if sar_preprocess in {"source_stats", "none"}:
        return sar
    source_mean = torch.as_tensor(stats["mean"][:2], device=sar.device, dtype=sar.dtype)[None, :, None, None]
    source_std = torch.as_tensor(stats["std"][:2], device=sar.device, dtype=sar.dtype)[None, :, None, None].clamp_min(1e-6)
    raw_sar = sar * source_std + source_mean
    if sar_preprocess == "target_x100_source_stats":
        return (raw_sar * 100.0 - source_mean) / source_std
    if sar_preprocess in {"target_image_zscore", "per_scene_zscore"}:
        mean = raw_sar.mean(dim=(-2, -1), keepdim=True)
        std = raw_sar.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (raw_sar - mean) / std
    if sar_preprocess == "clip_-35_5":
        return (raw_sar.clamp(-35.0, 5.0) - source_mean) / source_std
    if sar_preprocess == "clip_-30_0":
        return (raw_sar.clamp(-30.0, 0.0) - source_mean) / source_std
    raise ValueError(f"Unknown SAR preprocessing mode: {sar_preprocess}")


def _degrade_batch(
    batch: Dict[str, Any],
    condition: Dict[str, Any],
    base_seed: int,
    device: torch.device,
    sar_preprocess: str,
    stats: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    sar = batch["sar"].to(device, non_blocking=True)
    sar = _apply_sar_preprocess(sar, sar_preprocess, stats)
    opt = batch["opt"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)
    opt_mask = torch.ones((opt.shape[0], 1, opt.shape[-2], opt.shape[-1]), device=device, dtype=opt.dtype)
    opt_masked = opt.clone()
    ratio = float(condition["missing_ratio"]) / 100.0
    if ratio > 0:
        if condition["mask_type"] == "block":
            internal_mask_type = "random_block_mask"
        elif condition["mask_type"] == "cloud":
            internal_mask_type = "cloud_like_mask"
        else:
            internal_mask_type = "full_optical_missing"
        for i, chip_id in enumerate(batch["chip_id"]):
            seed = _stable_seed(chip_id, condition["condition"], base_seed)
            torch.manual_seed(seed)
            opt_masked[i], opt_mask[i, 0] = _torch_optical_missing(opt_masked[i], internal_mask_type, ratio)
    image = torch.cat([sar, opt_masked], dim=1)
    return {"sar": sar, "opt": opt_masked, "opt_mask": opt_mask, "image": image, "label": label, "valid_mask": valid_mask}


def _load_model(config: Dict[str, Any], model_name: str, checkpoint: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(model_name, config).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def _predict_logits(model: torch.nn.Module, model_name: str, batch: Dict[str, torch.Tensor], config: Dict[str, Any]) -> torch.Tensor:
    if model_name in {
        "mask_guided_late_fusion_unet",
        "mask_guided_late_fusion_sar_aux_unet",
        "mask_aware_cross_attention_fusion_unet",
        "mask_aware_cross_attention_no_sar_aux_unet",
        "mask_aware_cross_attention_fusion_unet_deep",
        "smagnet",
    }:
        output = model(batch["sar"], batch["opt"], batch["opt_mask"])
        return resolve_segmentation_logits(output, batch, model_inference_mode(config))
    if model_name == "s1_only_unet":
        return model(batch["sar"])
    if model_name == "s2_only_unet":
        return model(batch["opt"])
    return model(batch["image"])


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _save_prediction_tif(image_path: str, out_path: Path, pred: torch.Tensor, valid_mask: torch.Tensor) -> None:
    import rasterio

    ensure_dir(out_path.parent)
    arr = pred.squeeze().detach().cpu().to(torch.uint8).numpy()
    valid = valid_mask.squeeze().detach().cpu().numpy() > 0.5
    arr = arr.copy()
    arr[~valid] = 255
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


def _stretch_uint8(arr) -> Any:
    import numpy as np

    arr = arr.astype("float32")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return (out * 255).astype(np.uint8)


def _save_preview_png(
    image_path: str,
    out_path: Path,
    label: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw

    ensure_dir(out_path.parent)
    with rasterio.open(image_path) as src:
        raw = src.read().astype("float32")
    vv = _stretch_uint8(raw[0])
    rgb = np.stack([_stretch_uint8(raw[4]), _stretch_uint8(raw[3]), _stretch_uint8(raw[2])], axis=-1)
    gt = (label.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    pr = (pred.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    vm = (valid_mask.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    panels = [
        Image.fromarray(vv).convert("RGB"),
        Image.fromarray(rgb).convert("RGB"),
        Image.fromarray(gt).convert("RGB"),
        Image.fromarray(pr).convert("RGB"),
        Image.fromarray(vm).convert("RGB"),
    ]
    labels = ["SAR VV", "Optical RGB", "Ground Truth", "Prediction", "Valid Mask"]
    width, height = panels[0].size
    header = 24
    canvas = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, labels)):
        x = i * width
        canvas.paste(panel, (x, header))
        draw.text((x + 6, 5), title, fill=(0, 0, 0))
    canvas.save(out_path)


@torch.no_grad()
def evaluate_zero_shot(
    config: Dict[str, Any],
    data_root: str,
    checkpoint: str,
    stats_path: str,
    model_name: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    missing_ratios: Sequence[int],
    mask_types: Sequence[str],
    threshold: float,
    seed: int,
    save_predictions: bool,
    save_preview: bool,
    preview_limit_per_condition: int,
    sar_preprocess: str = "source_stats",
    max_batches: int | None = None,
    progress: bool = True,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = _load_stats(stats_path)
    ds = Sen1Floods116BandDataset(data_root, stats_path=stats_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate, pin_memory=torch.cuda.is_available())
    model = _load_model(config, model_name, checkpoint, device)
    conditions = _build_conditions(missing_ratios, mask_types)
    out_dir = ensure_dir(output_dir)

    overall_by_condition: Dict[str, ConfusionWithTN] = {c["condition"]: ConfusionWithTN() for c in conditions}
    per_chip_rows: List[Dict[str, Any]] = []

    condition_iter = tqdm(conditions, desc="zero-shot conditions", dynamic_ncols=True, disable=not progress)
    for condition in condition_iter:
        preview_count = 0
        condition_name = condition["condition"]
        total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
        batch_iter = tqdm(loader, total=total_batches, desc=condition_name, leave=False, dynamic_ncols=True, disable=not progress)
        for batch_idx, batch in enumerate(batch_iter):
            if max_batches is not None and batch_idx >= max_batches:
                break
            degraded = _degrade_batch(batch, condition, seed, device, sar_preprocess, stats)
            logits = _predict_logits(model, model_name, degraded, config)
            pred = (torch.sigmoid(logits) > threshold).float()
            for i, chip_id in enumerate(batch["chip_id"]):
                meter = ConfusionWithTN()
                meter.update_from_tensors(pred[i : i + 1], degraded["label"][i : i + 1], degraded["valid_mask"][i : i + 1])
                overall_by_condition[condition_name].merge(meter)
                valid_pixels = int((degraded["valid_mask"][i] > 0.5).sum().item())
                water_pixels = int(((degraded["label"][i] > 0.5) & (degraded["valid_mask"][i] > 0.5)).sum().item())
                per_chip_rows.append(
                    {
                        "chip_id": chip_id,
                        "missing_ratio": condition["missing_ratio"],
                        "mask_type": condition["mask_type"],
                        "condition": condition_name,
                        **meter.compute(),
                        "valid_pixels": valid_pixels,
                        "water_pixels": water_pixels,
                    }
                )
                if save_predictions:
                    pred_path = out_dir / "predictions" / f"{chip_id}_pred_{condition_name}.tif"
                    _save_prediction_tif(batch["image_path"][i], pred_path, pred[i], degraded["valid_mask"][i])
                if save_preview and preview_count < preview_limit_per_condition:
                    preview_path = out_dir / "preview" / f"{chip_id}_{condition_name}.png"
                    _save_preview_png(batch["image_path"][i], preview_path, degraded["label"][i], pred[i], degraded["valid_mask"][i])
                    preview_count += 1
            if progress:
                batch_iter.set_postfix(chips=len(per_chip_rows))

    by_ratio_rows: List[Dict[str, Any]] = []
    for condition in conditions:
        metrics = overall_by_condition[condition["condition"]].compute()
        by_ratio_rows.append(
            {
                "missing_ratio": condition["missing_ratio"],
                "mask_type": condition["mask_type"],
                "condition": condition["condition"],
                **metrics,
            }
        )

    overall = {
        "dataset": str(data_root),
        "num_samples": len(ds),
        "model": model_name,
        "checkpoint": str(checkpoint),
        "stats_path": str(stats_path),
        "sar_preprocess": sar_preprocess,
        "threshold": threshold,
        "conditions": by_ratio_rows,
    }
    (out_dir / "metrics_overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    _write_csv(
        out_dir / "metrics_by_ratio.csv",
        by_ratio_rows,
        ["missing_ratio", "mask_type", "condition", "iou", "f1", "precision", "recall", "tp", "fp", "fn", "tn"],
    )
    _write_csv(
        out_dir / "metrics_per_chip.csv",
        per_chip_rows,
        ["chip_id", "missing_ratio", "mask_type", "condition", "iou", "f1", "precision", "recall", "tp", "fp", "fn", "tn", "valid_pixels", "water_pixels"],
    )
    return overall


def _as_list(value: Any, default: Sequence[Any]) -> List[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot Sen1Floods11 evaluation for S1S2-Water models.")
    parser.add_argument("--config", default="configs/zero_shot/sen1floods11_proposed.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--missing-ratios", nargs="+", type=int, default=None)
    parser.add_argument("--mask-type", nargs="+", default=None, choices=["block", "cloud", "full"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--sar-preprocess",
        default=None,
        choices=["source_stats", "none", "target_x100_source_stats", "target_image_zscore", "per_scene_zscore", "clip_-35_5", "clip_-30_0"],
        help="Optional SAR-only preprocessing after loading source-stat-normalized tensors.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--no-save-preview", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--preview-limit-per-condition", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    data_root = args.data_root or config.get("dataset", {}).get("root")
    checkpoint = args.checkpoint or config.get("model", {}).get("checkpoint")
    stats_path = args.stats_path or config.get("normalization", {}).get("stats_path")
    model_name = args.model or config.get("model", {}).get("name", "mask_guided_late_fusion_unet")
    output_dir = args.output_dir or config.get("output", {}).get("dir", "outputs/zero_shot_sen1floods11/proposed")
    missing_ratios = _as_list(args.missing_ratios, eval_cfg.get("missing_ratios", [0, 25, 50, 75, 100]))
    mask_types = _as_list(args.mask_type, eval_cfg.get("mask_types", ["block", "cloud"]))
    if "full" not in mask_types:
        mask_types.append("full")
    if not data_root or not checkpoint or not stats_path:
        raise ValueError("--data-root, --checkpoint, and --stats-path are required via CLI or config")

    result = evaluate_zero_shot(
        config=config,
        data_root=data_root,
        checkpoint=checkpoint,
        stats_path=stats_path,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 4)),
        num_workers=int(args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers", 4)),
        missing_ratios=[int(v) for v in missing_ratios],
        mask_types=[str(v) for v in mask_types],
        threshold=float(args.threshold if args.threshold is not None else eval_cfg.get("threshold", 0.5)),
        seed=int(args.seed if args.seed is not None else eval_cfg.get("seed", 4)),
        save_predictions=bool(eval_cfg.get("save_predictions", True)) and not args.no_save_predictions,
        save_preview=bool(eval_cfg.get("save_preview", True)) and not args.no_save_preview,
        preview_limit_per_condition=int(args.preview_limit_per_condition if args.preview_limit_per_condition is not None else eval_cfg.get("preview_limit_per_condition", 20)),
        sar_preprocess=str(args.sar_preprocess or eval_cfg.get("sar_preprocess", "source_stats")),
        max_batches=args.max_batches,
        progress=not args.no_progress,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
