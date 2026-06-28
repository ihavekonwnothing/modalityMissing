from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.sen1floods11_6band import Sen1Floods116BandDataset
from models.factory import build_model
from models.output_utils import resolve_segmentation_logits
from train import _torch_optical_missing
from utils.config import ensure_dir, load_config


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.count += int(values.size)
        self.sum += float(values.sum(dtype=np.float64))
        self.sum_sq += float((values.astype(np.float64) ** 2).sum())
        self.min_value = min(self.min_value, float(values.min()))
        self.max_value = max(self.max_value, float(values.max()))

    def compute(self) -> Dict[str, float | int]:
        mean = self.sum / max(self.count, 1)
        var = self.sum_sq / max(self.count, 1) - mean * mean
        return {
            "count": self.count,
            "mean": mean,
            "std": float(np.sqrt(max(var, 0.0))),
            "min": self.min_value,
            "max": self.max_value,
        }


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom > 0 else 0.0


def _confusion_from_prob(prob: torch.Tensor, label: torch.Tensor, valid: torch.Tensor, threshold: float) -> Tuple[int, int, int, int]:
    pred = prob >= threshold
    target = label > 0.5
    mask = valid > 0.5
    tp = int((pred & target & mask).sum().item())
    fp = int((pred & ~target & mask).sum().item())
    fn = int((~pred & target & mask).sum().item())
    tn = int((~pred & ~target & mask).sum().item())
    return tp, fp, fn, tn


def _metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float | int]:
    return {
        "iou": _safe_div(tp, tp + fp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _load_model(config_path: str, model_name: str, checkpoint: str, device: torch.device):
    config = load_config(config_path)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(model_name, config).to(device)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def _collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["image", "sar", "opt", "label", "valid_mask"]
    out = {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}
    for k in ("chip_id", "image_path"):
        out[k] = [s[k] for s in samples]
    return out


def _predict(model, model_name: str, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if model_name == "s1_only_unet":
        return model(batch["sar"])
    if model_name in {"mask_guided_late_fusion_unet", "mask_aware_cross_attention_fusion_unet_deep"}:
        output = model(batch["sar"], batch["opt"], batch["opt_mask"])
        return resolve_segmentation_logits(output, batch, "adaptive_fallback")
    if model_name == "s2_only_unet":
        return model(batch["opt"])
    return model(batch["image"])


def _apply_sar_variant(batch: Dict[str, torch.Tensor], variant: str, source_mean: torch.Tensor, source_std: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = dict(batch)
    sar = out["sar"].clone()
    raw_sar = sar * source_std + source_mean
    if variant == "source_stats":
        out["sar"] = sar
    elif variant == "target_image_zscore":
        mean = raw_sar.mean(dim=(-2, -1), keepdim=True)
        std = raw_sar.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        out["sar"] = (raw_sar - mean) / std
    elif variant == "clip_-35_5":
        clipped = raw_sar.clamp(-35.0, 5.0)
        out["sar"] = (clipped - source_mean) / source_std
    elif variant == "clip_-30_0":
        clipped = raw_sar.clamp(-30.0, 0.0)
        out["sar"] = (clipped - source_mean) / source_std
    elif variant == "per_scene_zscore":
        mean = raw_sar.mean(dim=(-2, -1), keepdim=True)
        std = raw_sar.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        out["sar"] = (raw_sar - mean) / std
    elif variant == "target_x100_source_stats":
        scaled = raw_sar * 100.0
        out["sar"] = (scaled - source_mean) / source_std
    else:
        raise ValueError(f"Unknown SAR variant: {variant}")
    if "opt" in out:
        out["image"] = torch.cat([out["sar"], out["opt"]], dim=1)
    return out


@torch.no_grad()
def probability_diagnostics(args, out_dir: Path, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Sen1Floods116BandDataset(args.data_root, stats_path=args.stats_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate)
    models = [
        ("s1_only_unet", "full_100", args.s1_config, args.s1_checkpoint),
        ("mask_guided_late_fusion_unet", "full_100", args.proposed_config, args.proposed_checkpoint),
    ]
    source_mean = torch.tensor(stats["mean"][:2], device=device, dtype=torch.float32)[None, :, None, None]
    source_std = torch.tensor(stats["std"][:2], device=device, dtype=torch.float32)[None, :, None, None].clamp_min(1e-6)
    thresholds = [round(v, 2) for v in np.arange(0.05, 0.951, 0.05)]
    rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    variants = ["source_stats", "target_image_zscore", "clip_-35_5", "clip_-30_0", "per_scene_zscore", "target_x100_source_stats"]
    for model_name, condition, config_path, checkpoint in models:
        model = _load_model(config_path, model_name, checkpoint, device)
        for variant in variants:
            probs_for_quantile: List[torch.Tensor] = []
            prob_sum = 0.0
            valid_count = 0
            max_prob = 0.0
            pred_water_05 = 0
            conf_by_threshold = {t: [0, 0, 0, 0] for t in thresholds}
            for batch in loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                opt_missing = torch.zeros_like(batch["opt"])
                opt_mask = torch.zeros((batch["opt"].shape[0], 1, batch["opt"].shape[-2], batch["opt"].shape[-1]), device=device, dtype=batch["opt"].dtype)
                model_batch = {
                    "sar": batch["sar"],
                    "opt": opt_missing,
                    "opt_mask": opt_mask,
                    "image": torch.cat([batch["sar"], opt_missing], dim=1),
                    "label": batch["label"],
                    "valid_mask": batch["valid_mask"],
                }
                model_batch = _apply_sar_variant(model_batch, variant, source_mean, source_std)
                prob = torch.sigmoid(_predict(model, model_name, model_batch))
                valid = model_batch["valid_mask"] > 0.5
                valid_prob = prob[valid]
                if valid_prob.numel() == 0:
                    continue
                prob_sum += float(valid_prob.sum().item())
                valid_count += int(valid_prob.numel())
                max_prob = max(max_prob, float(valid_prob.max().item()))
                pred_water_05 += int((valid_prob >= 0.5).sum().item())
                flat_prob = valid_prob.detach().flatten()
                if flat_prob.numel() > args.quantile_samples_per_batch:
                    sample_idx = torch.linspace(0, flat_prob.numel() - 1, args.quantile_samples_per_batch, device=flat_prob.device).long()
                    flat_prob = flat_prob[sample_idx]
                probs_for_quantile.append(flat_prob.cpu())
                for t in thresholds:
                    tp, fp, fn, tn = _confusion_from_prob(prob, model_batch["label"], model_batch["valid_mask"], t)
                    conf = conf_by_threshold[t]
                    conf[0] += tp
                    conf[1] += fp
                    conf[2] += fn
                    conf[3] += tn
            q95 = float(torch.quantile(torch.cat(probs_for_quantile), 0.95).item()) if probs_for_quantile else 0.0
            best_t = 0.0
            best_metrics: Dict[str, Any] = {}
            for t, conf in conf_by_threshold.items():
                metrics = _metrics_from_confusion(*conf)
                threshold_rows.append({"model": model_name, "variant": variant, "threshold": t, **metrics})
                if not best_metrics or float(metrics["iou"]) > float(best_metrics["iou"]):
                    best_t = t
                    best_metrics = metrics
            rows.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "sar_variant": variant,
                    "mean_probability": _safe_div(prob_sum, valid_count),
                    "max_probability": max_prob,
                    "q95_probability": q95,
                    "predicted_water_pixels_threshold_0.5": pred_water_05,
                    "best_threshold": best_t,
                    "best_iou": best_metrics.get("iou", 0.0),
                    "best_f1": best_metrics.get("f1", 0.0),
                    "best_precision": best_metrics.get("precision", 0.0),
                    "best_recall": best_metrics.get("recall", 0.0),
                    "valid_pixels": valid_count,
                }
            )
    _write_csv(out_dir / "full_missing_probability_threshold_diagnostics.csv", rows)
    _write_csv(out_dir / "full_missing_threshold_sweep.csv", threshold_rows)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sar_distribution_diagnostics(args, out_dir: Path, stats: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import rasterio

    rng = np.random.default_rng(args.seed)
    source_mean = np.asarray(stats["mean"][:2], dtype=np.float32)[:, None, None]
    source_std = np.asarray(stats["std"][:2], dtype=np.float32)[:, None, None]
    source_stats = {("source_train", "VV", "all"): RunningStats(), ("source_train", "VH", "all"): RunningStats()}
    source_label_stats = {
        ("source_train", "VV", "water"): RunningStats(),
        ("source_train", "VV", "non_water"): RunningStats(),
        ("source_train", "VH", "water"): RunningStats(),
        ("source_train", "VH", "non_water"): RunningStats(),
    }
    source_samples = {"VV": [], "VH": [], "VV_water": [], "VV_non_water": [], "VH_water": [], "VH_non_water": []}
    manifest_rows = []
    with Path(args.s1s2_cache_dir, "manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest_rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    if args.max_source_patches and len(manifest_rows) > args.max_source_patches:
        idx = rng.choice(len(manifest_rows), size=args.max_source_patches, replace=False)
        manifest_rows = [manifest_rows[i] for i in idx]
    for row in manifest_rows:
        with np.load(Path(args.s1s2_cache_dir) / row["path"]) as data:
            sar = data["sar"].astype(np.float32) * source_std + source_mean
            mask = data["mask"].astype(bool)
            valid = data["valid_mask"].astype(bool)
        for ci, band in enumerate(["VV", "VH"]):
            vals = sar[ci][valid]
            source_stats[("source_train", band, "all")].update(vals)
            source_label_stats[("source_train", band, "water")].update(sar[ci][valid & mask])
            source_label_stats[("source_train", band, "non_water")].update(sar[ci][valid & ~mask])
            if vals.size:
                source_samples[band].append(rng.choice(vals, size=min(args.hist_samples_per_patch, vals.size), replace=False))
            water_vals = sar[ci][valid & mask]
            non_vals = sar[ci][valid & ~mask]
            if water_vals.size:
                source_samples[f"{band}_water"].append(rng.choice(water_vals, size=min(args.hist_samples_per_patch, water_vals.size), replace=False))
            if non_vals.size:
                source_samples[f"{band}_non_water"].append(rng.choice(non_vals, size=min(args.hist_samples_per_patch, non_vals.size), replace=False))

    target_stats = {("sen1floods11", "VV", "all"): RunningStats(), ("sen1floods11", "VH", "all"): RunningStats()}
    target_label_stats = {
        ("sen1floods11", "VV", "water"): RunningStats(),
        ("sen1floods11", "VV", "non_water"): RunningStats(),
        ("sen1floods11", "VH", "water"): RunningStats(),
        ("sen1floods11", "VH", "non_water"): RunningStats(),
    }
    target_samples = {"VV": [], "VH": [], "VV_water": [], "VV_non_water": [], "VH_water": [], "VH_non_water": []}
    meta = list(csv.DictReader(Path(args.data_root, "metadata.csv").open(newline="", encoding="utf-8")))
    for row in meta:
        with rasterio.open(Path(args.data_root) / row["image"]) as src:
            sar = src.read(indexes=[1, 2]).astype(np.float32)
        with rasterio.open(Path(args.data_root) / row["label"]) as src:
            label = src.read(1) > 0
        with rasterio.open(Path(args.data_root) / row["valid_mask"]) as src:
            valid = src.read(1) > 0
        for ci, band in enumerate(["VV", "VH"]):
            vals = sar[ci][valid]
            target_stats[("sen1floods11", band, "all")].update(vals)
            target_label_stats[("sen1floods11", band, "water")].update(sar[ci][valid & label])
            target_label_stats[("sen1floods11", band, "non_water")].update(sar[ci][valid & ~label])
            if vals.size:
                target_samples[band].append(rng.choice(vals, size=min(args.hist_samples_per_patch, vals.size), replace=False))
            water_vals = sar[ci][valid & label]
            non_vals = sar[ci][valid & ~label]
            if water_vals.size:
                target_samples[f"{band}_water"].append(rng.choice(water_vals, size=min(args.hist_samples_per_patch, water_vals.size), replace=False))
            if non_vals.size:
                target_samples[f"{band}_non_water"].append(rng.choice(non_vals, size=min(args.hist_samples_per_patch, non_vals.size), replace=False))

    stat_rows = []
    for key, meter in {**source_stats, **target_stats, **source_label_stats, **target_label_stats}.items():
        dataset, band, group = key
        stat_rows.append({"dataset": dataset, "band": band, "group": group, **meter.compute()})
    _write_csv(out_dir / "sar_distribution_stats.csv", stat_rows)

    hist_rows = []
    for band in ["VV", "VH"]:
        src_vals = np.concatenate(source_samples[band]) if source_samples[band] else np.array([], dtype=np.float32)
        tgt_vals = np.concatenate(target_samples[band]) if target_samples[band] else np.array([], dtype=np.float32)
        lo = float(np.nanpercentile(np.concatenate([src_vals, tgt_vals]), 0.5))
        hi = float(np.nanpercentile(np.concatenate([src_vals, tgt_vals]), 99.5))
        bins = np.linspace(lo, hi, args.hist_bins + 1)
        for dataset, vals in [("source_train", src_vals), ("sen1floods11", tgt_vals)]:
            counts, edges = np.histogram(vals, bins=bins)
            total = max(int(counts.sum()), 1)
            for i, count in enumerate(counts):
                hist_rows.append({"band": band, "dataset": dataset, "bin_left": edges[i], "bin_right": edges[i + 1], "count": int(count), "density": float(count / total)})
        plt.figure(figsize=(7, 4))
        plt.hist(src_vals, bins=bins, density=True, alpha=0.5, label="S1S2-Water train")
        plt.hist(tgt_vals, bins=bins, density=True, alpha=0.5, label="Sen1Floods11")
        plt.title(f"{band} raw SAR distribution")
        plt.xlabel("Raw SAR value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"sar_hist_{band}.png", dpi=160)
        plt.close()
        plt.figure(figsize=(8, 4))
        for dataset, samples in [("source", source_samples), ("target", target_samples)]:
            for group, linestyle in [("water", "-"), ("non_water", "--")]:
                vals = np.concatenate(samples[f"{band}_{group}"]) if samples[f"{band}_{group}"] else np.array([], dtype=np.float32)
                if vals.size:
                    plt.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, linestyle=linestyle, label=f"{dataset} {group}")
        plt.title(f"{band} water vs non-water SAR distribution")
        plt.xlabel("Raw SAR value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"sar_water_nonwater_hist_{band}.png", dpi=160)
        plt.close()
    _write_csv(out_dir / "sar_histograms.csv", hist_rows)


@torch.no_grad()
def qualitative_comparison(args, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import rasterio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(out_dir / "qualitative_s2_vs_proposed")
    ds = Sen1Floods116BandDataset(args.data_root, stats_path=args.stats_path)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate)
    s2_model = _load_model(args.s2_config, "s2_only_unet", args.s2_checkpoint, device)
    prop_model = _load_model(args.proposed_config, "mask_guided_late_fusion_unet", args.proposed_checkpoint, device)
    rows = []
    made = 0
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        opt_mask = torch.ones((batch["opt"].shape[0], 1, batch["opt"].shape[-2], batch["opt"].shape[-1]), device=device)
        prop_batch = {"sar": batch["sar"], "opt": batch["opt"], "opt_mask": opt_mask, "image": batch["image"], "label": batch["label"], "valid_mask": batch["valid_mask"]}
        s2_prob = torch.sigmoid(s2_model(batch["opt"]))
        prop_prob = torch.sigmoid(_predict(prop_model, "mask_guided_late_fusion_unet", prop_batch))
        s2_pred = s2_prob >= 0.5
        prop_pred = prop_prob >= 0.5
        valid = batch["valid_mask"] > 0.5
        label = batch["label"] > 0.5
        for i, chip_id in enumerate(batch["chip_id"]):
            s2_water = int((s2_pred[i] & valid[i]).sum().item())
            prop_water = int((prop_pred[i] & valid[i]).sum().item())
            gt_water = int((label[i] & valid[i]).sum().item())
            rows.append({"chip_id": chip_id, "gt_water_pixels": gt_water, "s2_pred_water_pixels": s2_water, "proposed_pred_water_pixels": prop_water, "s2_minus_proposed": s2_water - prop_water})
        if made >= args.preview_count:
            continue
        for i, chip_id in enumerate(batch["chip_id"]):
            if made >= args.preview_count:
                break
            with rasterio.open(batch["image_path"][i]) as src:
                raw = src.read().astype(np.float32)
            rgb = np.stack([_stretch(raw[4]), _stretch(raw[3]), _stretch(raw[2])], axis=-1)
            vv = _stretch(raw[0])
            gt = label[i, 0].detach().cpu().numpy()
            s2p = s2_pred[i, 0].detach().cpu().numpy()
            pp = prop_pred[i, 0].detach().cpu().numpy()
            diff = np.zeros((*gt.shape, 3), dtype=np.float32)
            diff[s2p & ~pp] = [1, 0, 0]
            diff[pp & ~s2p] = [0, 0, 1]
            diff[s2p & pp] = [0, 1, 0]
            panels = [(vv, "SAR VV"), (rgb, "Optical RGB"), (gt, "GT"), (s2p, "S2-only pred"), (pp, "Proposed pred"), (diff, "Diff R=S2 only B=Prop only")]
            fig, axes = plt.subplots(1, len(panels), figsize=(18, 3))
            for ax, (img, title) in zip(axes, panels):
                ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
                ax.set_title(title)
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / "qualitative_s2_vs_proposed" / f"{chip_id}.png", dpi=160)
            plt.close(fig)
            made += 1
    _write_csv(out_dir / "s2_vs_proposed_pred_area.csv", rows)


def _stretch(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1
    return np.nan_to_num(np.clip((arr - lo) / (hi - lo), 0, 1), nan=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="transfer_dataset/Sen1Floods11_6band")
    parser.add_argument("--stats-path", default="data/s1s2_water_patch_cache_512/stats.json")
    parser.add_argument("--s1s2-cache-dir", default="data/s1s2_water_patch_cache_512")
    parser.add_argument("--output-dir", default="outputs/zero_shot_sen1floods11/diagnostics")
    parser.add_argument("--s1-config", default="configs/s1s2_water/baseline_s1_unet_effb0.yaml")
    parser.add_argument("--s1-checkpoint", default="outputs/s1s2_water/baselines/s1_unet_effb0_scheduler_v2/checkpoints/best.ckpt")
    parser.add_argument("--s2-config", default="configs/s1s2_water/baseline_s2_unet_effb0.yaml")
    parser.add_argument("--s2-checkpoint", default="outputs/s1s2_water/baselines/s2_unet_effb0_scheduler_v2/checkpoints/best.ckpt")
    parser.add_argument("--proposed-config", default="configs/zero_shot/sen1floods11_proposed.yaml")
    parser.add_argument("--proposed-checkpoint", default="outputs/s1s2_water/proposed/mask_guided_late_fusion_unet_ddp/checkpoints/best.ckpt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-source-patches", type=int, default=2048)
    parser.add_argument("--hist-samples-per-patch", type=int, default=512)
    parser.add_argument("--hist-bins", type=int, default=80)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--quantile-samples-per-batch", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    out_dir = ensure_dir(args.output_dir)
    stats = json.loads(Path(args.stats_path).read_text(encoding="utf-8"))
    prob_rows = probability_diagnostics(args, out_dir, stats)
    sar_distribution_diagnostics(args, out_dir, stats)
    qualitative_comparison(args, out_dir)
    (out_dir / "diagnostic_summary.json").write_text(json.dumps({"probability_rows": prob_rows, "output_dir": str(out_dir)}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "probability_rows": prob_rows}, indent=2))


if __name__ == "__main__":
    main()
