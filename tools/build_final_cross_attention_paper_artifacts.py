from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from datasets.s1s2_water_cache import S1S2WaterPatchCacheDataset
from datasets.sen1floods11_6band import Sen1Floods116BandDataset
from models.factory import build_model, select_model_input
from models.output_utils import model_inference_mode, resolve_segmentation_logits
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config


OUT_DIR = Path("outputs/final_paper_artifacts_cross_attention")
METRIC_DIR = Path("outputs/final_metric_summary_2026_06_24/with_cross_attention")


MODEL_SPECS = {
    "s1_only_unet": {
        "label": "S1-only",
        "model_name": "s1_only_unet",
        "config": "configs/s1s2_water/baseline_s1_unet_effb0.yaml",
        "checkpoint": "outputs/s1s2_water/baselines/s1_unet_effb0_scheduler_v2/checkpoints/best.ckpt",
    },
    "late_fusion_unet_robust": {
        "label": "Late Fusion + Missing Training",
        "model_name": "late_fusion_unet",
        "config": "configs/s1s2_water/baseline_late_fusion_unet_robust_ddp.yaml",
        "checkpoint": "outputs/s1s2_water/robust_baselines/late_fusion_unet_robust_ddp/checkpoints/best.ckpt",
    },
    "smagnet": {
        "label": "SMAGNet",
        "model_name": "smagnet",
        "config": "configs/s1s2_water/baseline_smagnet.yaml",
        "checkpoint": "outputs/s1s2_water/baselines/smagnet_ddp/checkpoints/best.ckpt",
    },
    "mask_guided_late_fusion_unet": {
        "label": "Mask-Guided",
        "model_name": "mask_guided_late_fusion_unet",
        "config": "configs/s1s2_water/proposed_mask_guided_late_fusion_unet_ddp.yaml",
        "checkpoint": "outputs/s1s2_water/proposed/mask_guided_late_fusion_unet_ddp/checkpoints/best.ckpt",
    },
    "mask_aware_cross_attention_fusion_unet": {
        "label": "Ours",
        "model_name": "mask_aware_cross_attention_fusion_unet",
        "config": "configs/s1s2_water/mask_aware_cross_attention_fusion.yaml",
        "checkpoint": "outputs/s1s2_water/proposed/mask_aware_cross_attention_fusion_unet_ddp/checkpoints/best.ckpt",
    },
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return ""


def _stretch(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _stretch_sar(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _denormalize(arr: np.ndarray, stats: Dict[str, Any], offset: int, count: int) -> np.ndarray:
    mean = np.asarray(stats["mean"][offset : offset + count], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"][offset : offset + count], dtype=np.float32)[:, None, None]
    return arr.astype(np.float32) * std + mean


def _rgb_from_opt(opt: np.ndarray) -> np.ndarray:
    rgb = np.stack([opt[2], opt[1], opt[0]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        out[..., c] = _stretch(rgb[..., c])
    return out


def _binary_iou(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    pred_b = pred > 0.5
    target_b = target > 0.5
    valid_b = valid > 0.5
    inter = (pred_b & target_b & valid_b).sum().item()
    union = ((pred_b | target_b) & valid_b).sum().item()
    return float(inter) / float(union) if union else 0.0


def _load_model(model_id: str, device: torch.device) -> tuple[torch.nn.Module, Dict[str, Any]]:
    spec = MODEL_SPECS[model_id]
    config = load_config(spec["config"])
    ckpt = torch.load(spec["checkpoint"], map_location="cpu")
    model = build_model(spec.get("model_name", model_id), config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def _predict(model: torch.nn.Module, model_id: str, config: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    model_name = MODEL_SPECS.get(model_id, {}).get("model_name", model_id)
    output = model(select_model_input(batch, model_name))
    logits = resolve_segmentation_logits(output, batch, model_inference_mode(config))
    return (torch.sigmoid(logits) >= 0.5).float()


def _full_missing_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    out["opt"] = torch.zeros_like(out["opt"])
    out["opt_mask"] = torch.zeros((out["opt"].shape[0], 1, out["opt"].shape[-2], out["opt"].shape[-1]), device=device, dtype=out["opt"].dtype)
    out["image"] = torch.cat([out["sar"], out["opt"]], dim=1)
    return out


def _apply_transfer_sar_x100(batch: Dict[str, torch.Tensor], stats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out = dict(batch)
    mean = torch.as_tensor(stats["mean"][:2], device=out["sar"].device, dtype=out["sar"].dtype)[None, :, None, None]
    std = torch.as_tensor(stats["std"][:2], device=out["sar"].device, dtype=out["sar"].dtype)[None, :, None, None].clamp_min(1e-6)
    raw_sar = out["sar"] * std + mean
    out["sar"] = (raw_sar * 100.0 - mean) / std
    out["image"] = torch.cat([out["sar"], out["opt"]], dim=1)
    return out


def _plot_rows(
    rows: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    pred_columns: Iterable[str] = tuple(MODEL_SPECS.keys()),
) -> None:
    labels = ["SAR VV", "Optical RGB", "GT", *[MODEL_SPECS[c]["label"] for c in pred_columns]]
    nrows = len(rows)
    ncols = len(labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols, 2.05 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])
    for r_idx, row in enumerate(rows):
        sar = row["sar"]
        opt = row["opt"]
        gt = row["gt"]
        panels = [_stretch_sar(row.get("sar_raw", sar)[0]), _rgb_from_opt(row.get("opt_raw", opt)), gt[0]]
        cmaps = ["gray", None, "Blues"]
        for model_id in pred_columns:
            panels.append(row["preds"][model_id][0])
            cmaps.append("Blues")
        for c_idx, (panel, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[r_idx, c_idx]
            ax.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.axis("off")
            if r_idx == 0:
                ax.set_title(labels[c_idx], fontsize=10)
            if c_idx == 0:
                ax.text(0.02, 0.96, row["name"], transform=ax.transAxes, va="top", ha="left", fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.55, "pad": 1})
    fig.suptitle(title, fontsize=13)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _save_panel(path: Path, panel: np.ndarray, cmap: str | None = None) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    ax.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_individual_rows(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    pred_columns: Iterable[str] = tuple(MODEL_SPECS.keys()),
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for row in rows:
        sample_dir = out_dir / _safe_name(row["name"])
        _save_panel(sample_dir / "sar_vv.png", _stretch_sar(row.get("sar_raw", row["sar"])[0]), cmap="gray")
        _save_panel(sample_dir / "optical_rgb.png", _rgb_from_opt(row.get("opt_raw", row["opt"])))
        _save_panel(sample_dir / "gt.png", row["gt"][0], cmap="Blues")
        for model_id in pred_columns:
            filename = f"pred_{model_id}.png"
            _save_panel(sample_dir / filename, row["preds"][model_id][0], cmap="Blues")


def write_story_markdown() -> None:
    ensure_dir(OUT_DIR)
    s1 = _read_csv(METRIC_DIR / "s1s2_water_clean_full_with_cross_attention.csv")
    tr = _read_csv(METRIC_DIR / "sen1floods11_sar_x100_clean_full_with_cross_attention.csv")
    s1_iou = _read_csv(METRIC_DIR / "s1s2_water_iou_pivot.csv")
    tr_iou = _read_csv(METRIC_DIR / "sen1floods11_sar_x100_iou_pivot.csv")
    all_rows = _read_csv(METRIC_DIR / "all_metrics_with_cross_attention_long.csv")

    def compact_table(title: str, rows: List[Dict[str, str]]) -> List[str]:
        lines = [f"## {title}\n", "| model | training | clean IoU | full missing IoU |\n", "|---|---|---:|---:|\n"]
        for r in rows:
            if r["model_id"] == "mask_aware_cross_attention_fusion_unet_deep":
                continue
            lines.append(f"| {r['model']} | {r['training']} | {_fmt(r.get('clean_IoU'))} | {_fmt(r.get('full_missing_100_IoU'))} |\n")
        lines.append("\n")
        return lines

    def iou_table(title: str, rows: List[Dict[str, str]]) -> List[str]:
        if not rows:
            return []
        fields = [k for k in rows[0].keys() if k not in {"model_id", "model", "training"}]
        lines = [f"## {title}\n", "| model | " + " | ".join(fields) + " |\n", "|---|" + "|".join(["---:"] * len(fields)) + "|\n"]
        for r in rows:
            if r["model_id"] == "mask_aware_cross_attention_fusion_unet_deep":
                continue
            lines.append("| " + r["model"] + " | " + " | ".join(_fmt(r.get(f)) for f in fields) + " |\n")
        lines.append("\n")
        return lines

    def final_model_metric_table(title: str, dataset: str, experiment: str) -> List[str]:
        rows = [
            r
            for r in all_rows
            if r.get("model_id") == "mask_aware_cross_attention_fusion_unet"
            and r.get("dataset") == dataset
            and r.get("experiment") == experiment
        ]
        order = {
            "clean": 0,
            "random_block_25": 1,
            "random_block_50": 2,
            "random_block_75": 3,
            "cloud_like_25": 4,
            "cloud_like_50": 5,
            "cloud_like_75": 6,
            "full_missing_100": 7,
            "block_25": 1,
            "block_50": 2,
            "block_75": 3,
            "cloud_25": 4,
            "cloud_50": 5,
            "cloud_75": 6,
            "full_100": 7,
        }
        rows = sorted(rows, key=lambda r: order.get(r.get("condition_id", ""), 99))
        lines = [f"## {title}\n", "| condition | IoU | F1 | Precision | Recall |\n", "|---|---:|---:|---:|---:|\n"]
        for r in rows:
            lines.append(f"| {r['condition']} | {_fmt(r['IoU'])} | {_fmt(r['F1'])} | {_fmt(r['Precision'])} | {_fmt(r['Recall'])} |\n")
        lines.append("\n")
        return lines

    md: List[str] = []
    md.append("# Final Model Metrics and Paper Story\n\n")
    md.append("Final model: **Mask-Aware Cross-Attention Fusion U-Net**. Cross-attention deep is intentionally excluded from this summary.\n\n")
    md.append("## Paper Story Outline\n")
    md.append("1. **Problem.** SAR-optical water segmentation is usually evaluated when optical imagery is available, but in flood and water mapping scenarios optical observations are often partially or fully missing.\n")
    md.append("2. **Observation.** Clean-trained optical or fusion baselines can perform well on clean S1S2-Water, but collapse under full optical missing input.\n")
    md.append("3. **Protocol.** We evaluate controlled optical degradation with clean, block missing, cloud-like missing, and full optical missing conditions, plus zero-shot Sen1Floods11 transfer with SAR x100 normalization.\n")
    md.append("4. **Method.** The final model uses dual encoders, mask-guided fusion at high-resolution stages, mask-aware cross-attention at low-resolution stages, and an auxiliary SAR segmentation head.\n")
    md.append("5. **Key result.** The final model is not the best clean-optical model, but it is the strongest severe-missing model: full-missing IoU reaches 0.8065 on S1S2-Water and 0.5217 on Sen1Floods11 transfer.\n")
    md.append("6. **Conclusion.** Missing-modality robustness requires both controlled missing-modality training and an explicit SAR fallback path; the SAR auxiliary branch improves complete optical absence and cross-dataset robustness.\n\n")
    md.extend(compact_table("S1S2-Water Clean / Full Missing", s1))
    md.extend(compact_table("Sen1Floods11 SAR x100 Transfer Clean / Full Missing", tr))
    md.extend(final_model_metric_table("Final Model All Metrics on S1S2-Water", "S1S2-Water", "degradation_test_scene31_excluded"))
    md.extend(final_model_metric_table("Final Model All Metrics on Sen1Floods11 SAR x100 Transfer", "Sen1Floods11", "zero_shot_transfer_sar_x100_source_stats"))
    md.extend(iou_table("S1S2-Water All Conditions IoU", s1_iou))
    md.extend(iou_table("Sen1Floods11 SAR x100 Transfer All Conditions IoU", tr_iou))
    md.append("## Figure Outputs\n")
    md.append("- `figures/s1s2_water_full_missing_top6.png`: S1S2-Water full optical missing qualitative comparison.\n")
    md.append("- `figures/sen1floods11_full_missing_top6.png`: Sen1Floods11 full optical missing transfer qualitative comparison.\n")
    md.append("- Each row is one selected sample where the final model performs well; columns are SAR VV, Optical RGB, GT, S1-only, Late Fusion + Missing Training, Mask-Guided, and Ours.\n")
    md.append("- `qualitative_tiles/`: the same selected samples exported as separate PNG files per tile/chip, without layout composition.\n")
    (OUT_DIR / "paper_story_and_metrics.md").write_text("".join(md), encoding="utf-8")


@torch.no_grad()
def build_s1s2_qualitative(args: argparse.Namespace, device: torch.device) -> None:
    ds = S1S2WaterPatchCacheDataset(args.s1s2_cache_dir, "test", exclude_scenes=[31])
    stats = ds.cache_metadata.get("stats") or {}
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=segmentation_collate, num_workers=0)
    ours, ours_cfg = _load_model("mask_aware_cross_attention_fusion_unet", device)
    late, late_cfg = _load_model("late_fusion_unet_robust", device)
    smagnet, smagnet_cfg = _load_model("smagnet", device)
    candidates: List[tuple[float, int, int, float, float]] = []
    seen = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="select S1S2 examples", dynamic_ncols=True)):
        if seen >= args.max_s1s2_scan:
            break
        full = _full_missing_batch(batch, device)
        ours_pred = _predict(ours, "mask_aware_cross_attention_fusion_unet", ours_cfg, full)
        late_pred = _predict(late, "late_fusion_unet_robust", late_cfg, full)
        smagnet_pred = _predict(smagnet, "smagnet", smagnet_cfg, full)
        for i in range(ours_pred.shape[0]):
            water = int(((full["mask"][i] > 0.5) & (full["valid_mask"][i] > 0.5)).sum().item())
            if water < args.min_water_pixels:
                continue
            ours_iou = _binary_iou(ours_pred[i : i + 1], full["mask"][i : i + 1], full["valid_mask"][i : i + 1])
            late_iou = _binary_iou(late_pred[i : i + 1], full["mask"][i : i + 1], full["valid_mask"][i : i + 1])
            smagnet_iou = _binary_iou(smagnet_pred[i : i + 1], full["mask"][i : i + 1], full["valid_mask"][i : i + 1])
            strongest_baseline_iou = max(late_iou, smagnet_iou)
            margin = ours_iou - strongest_baseline_iou
            score = ours_iou + 0.25 * max(margin, 0.0) - 0.75 * max(-margin, 0.0)
            candidates.append((score, batch_idx * args.batch_size + i, water, ours_iou, strongest_baseline_iou))
        seen += ours_pred.shape[0]
    selected = [idx for _, idx, _, _, _ in sorted(candidates, reverse=True)[: args.num_examples]]
    if not selected:
        raise RuntimeError("No S1S2 qualitative candidates found. Increase --max-s1s2-scan or lower --min-water-pixels.")

    models = {}
    cfgs = {}
    for model_id in MODEL_SPECS:
        models[model_id], cfgs[model_id] = _load_model(model_id, device)
    rows = []
    for idx in tqdm(selected, desc="render S1S2 examples", dynamic_ncols=True):
        sample = ds[idx]
        batch = segmentation_collate([sample])
        full = _full_missing_batch(batch, device)
        preds = {model_id: _predict(models[model_id], model_id, cfgs[model_id], full)[0].cpu().numpy() for model_id in MODEL_SPECS}
        rows.append(
            {
                "name": f"{sample['sample_id']}:{sample['metadata']['tile']['x']},{sample['metadata']['tile']['y']}",
                "sar": sample["sar"].numpy(),
                "sar_raw": _denormalize(sample["sar"].numpy(), stats, 0, 2) if stats else sample["sar"].numpy(),
                "opt": sample["opt"].numpy(),
                "opt_raw": _denormalize(sample["opt"].numpy(), stats, 2, 4) if stats else sample["opt"].numpy(),
                "gt": sample["mask"].numpy(),
                "preds": preds,
            }
        )
    _plot_rows(rows, OUT_DIR / "figures" / "s1s2_water_full_missing_top6.png", "S1S2-Water: full optical missing qualitative comparison")
    _save_individual_rows(rows, OUT_DIR / "qualitative_tiles" / "s1s2_water_full_missing")


@torch.no_grad()
def build_transfer_qualitative(args: argparse.Namespace, device: torch.device) -> None:
    stats_path = Path(args.stats_path)
    import json

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    ds = Sen1Floods116BandDataset(args.transfer_root, stats_path=stats_path)
    per_chip = _read_csv(Path("outputs/zero_shot_sen1floods11_sar_x100/mask_aware_cross_attention_fusion_unet/metrics_per_chip.csv"))
    late_per_chip = _read_csv(Path("outputs/zero_shot_sen1floods11_sar_x100/late_fusion_unet_robust/metrics_per_chip.csv"))
    smagnet_per_chip = _read_csv(Path("outputs/zero_shot_sen1floods11/smagnet_best/metrics_per_chip.csv"))
    late_iou_by_chip = {r["chip_id"]: float(r["iou"]) for r in late_per_chip if r["condition"] == "full_100"}
    smagnet_iou_by_chip = {r["chip_id"]: float(r["iou"]) for r in smagnet_per_chip if r["condition"] == "full_100"}
    selected_ids = [
        r["chip_id"]
        for r in sorted(
            [r for r in per_chip if r["condition"] == "full_100" and int(r["water_pixels"]) >= args.min_water_pixels],
            key=lambda r: float(r["iou"])
            + 0.25
            * max(
                float(r["iou"]) - max(late_iou_by_chip.get(r["chip_id"], 0.0), smagnet_iou_by_chip.get(r["chip_id"], 0.0)),
                0.0,
            )
            - 0.75
            * max(
                max(late_iou_by_chip.get(r["chip_id"], 0.0), smagnet_iou_by_chip.get(r["chip_id"], 0.0)) - float(r["iou"]),
                0.0,
            ),
            reverse=True,
        )[: args.num_examples]
    ]
    if not selected_ids:
        raise RuntimeError("No transfer qualitative candidates found. Lower --min-water-pixels or check metrics_per_chip.csv.")
    index_by_chip = {row["chip_id"]: i for i, row in enumerate(ds.rows)}
    models = {}
    cfgs = {}
    for model_id in MODEL_SPECS:
        models[model_id], cfgs[model_id] = _load_model(model_id, device)
    rows = []
    for chip_id in tqdm(selected_ids, desc="render transfer examples", dynamic_ncols=True):
        sample = ds[index_by_chip[chip_id]]
        batch = {
            "image": sample["image"][None],
            "sar": sample["sar"][None],
            "opt": sample["opt"][None],
            "label": sample["label"][None],
            "valid_mask": sample["valid_mask"][None],
        }
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = _apply_transfer_sar_x100(batch, stats)
        batch["mask"] = batch["label"]
        full = _full_missing_batch(batch, device)
        preds = {model_id: _predict(models[model_id], model_id, cfgs[model_id], full)[0].cpu().numpy() for model_id in MODEL_SPECS}
        rows.append(
            {
                "name": chip_id,
                "sar": sample["sar"].numpy(),
                "sar_raw": _denormalize(sample["sar"].numpy(), stats, 0, 2),
                "opt": sample["opt"].numpy(),
                "opt_raw": _denormalize(sample["opt"].numpy(), stats, 2, 4),
                "gt": sample["label"].numpy(),
                "preds": preds,
            }
        )
    _plot_rows(rows, OUT_DIR / "figures" / "sen1floods11_full_missing_top6.png", "Sen1Floods11 transfer: full optical missing qualitative comparison")
    _save_individual_rows(rows, OUT_DIR / "qualitative_tiles" / "sen1floods11_full_missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-s1s2-scan", type=int, default=256)
    parser.add_argument("--min-water-pixels", type=int, default=1000)
    parser.add_argument("--s1s2-cache-dir", default="data/s1s2_water_patch_cache_512")
    parser.add_argument("--transfer-root", default="transfer_dataset/Sen1Floods11_6band")
    parser.add_argument("--stats-path", default="data/s1s2_water_patch_cache_512/stats.json")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    ensure_dir(OUT_DIR / "figures")
    write_story_markdown()
    if args.skip_figures:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_s1s2_qualitative(args, device)
    build_transfer_qualitative(args, device)


if __name__ == "__main__":
    main()
