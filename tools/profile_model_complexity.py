from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.factory import build_model, select_model_input
from models.mask_aware_cross_attention_fusion_unet import MaskAwareCrossAttentionBlock
from models.mask_aware_cross_attention_fusion_unet_deep import MaskAwareCrossAttentionBlock as MaskAwareCrossAttentionBlockDeep
from models.output_utils import model_inference_mode, resolve_segmentation_logits
from train import resolve_model_name
from utils.config import ensure_dir, load_config


MODEL_SPECS = [
    ("s1_only_unet", "S1-only U-Net", "configs/s1s2_water/baseline_s1_unet_effb0.yaml", "EfficientNet-B0 SMP"),
    ("s2_only_unet", "S2-only U-Net", "configs/s1s2_water/baseline_s2_unet_effb0.yaml", "EfficientNet-B0 SMP"),
    ("early_fusion_unet", "Early Fusion U-Net", "configs/s1s2_water/baseline_early_fusion_unet.yaml", "custom lightweight U-Net"),
    ("late_fusion_unet", "Late Fusion U-Net", "configs/s1s2_water/baseline_late_fusion_unet.yaml", "custom lightweight U-Net"),
    ("early_fusion_unet", "Early Fusion U-Net + Missing Training", "configs/s1s2_water/baseline_early_fusion_unet_robust_ddp.yaml", "same architecture as Early Fusion"),
    ("late_fusion_unet", "Late Fusion U-Net + Missing Training", "configs/s1s2_water/baseline_late_fusion_unet_robust_ddp.yaml", "same architecture as Late Fusion"),
    ("mask_guided_late_fusion_unet", "Mask-Guided Late Fusion U-Net", "configs/s1s2_water/proposed_mask_guided_late_fusion_unet_ddp.yaml", "mask-guided fusion"),
    ("mask_guided_late_fusion_unet", "Mask-Guided Late Fusion U-Net Clean-only", "configs/s1s2_water/proposed_mask_guided_late_fusion_unet_clean_only_ddp.yaml", "same architecture as Mask-Guided"),
    ("mask_aware_cross_attention_fusion_unet", "Ours: Cross-Attn + SAR Aux", "configs/s1s2_water/mask_aware_cross_attention_fusion.yaml", "final model"),
    ("mask_aware_cross_attention_no_sar_aux_unet", "Ablation: Cross-Attn only", "configs/s1s2_water/ablation_cross_attention_no_sar_aux.yaml", "no SAR auxiliary head"),
    ("mask_guided_late_fusion_sar_aux_unet", "Ablation: SAR Aux only", "configs/s1s2_water/ablation_mask_guided_sar_aux.yaml", "no cross-attention"),
    ("mask_aware_cross_attention_fusion_unet_deep", "Cross-Attn Deep", "configs/s1s2_water/mask_aware_cross_attention_fusion_unet_deep.yaml", "experimental deep variant"),
]


def _count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _make_batch(model_name: str, size: int, device: torch.device) -> dict[str, torch.Tensor]:
    sar = torch.randn(1, 2, size, size, device=device)
    opt = torch.randn(1, 4, size, size, device=device)
    opt_mask = torch.ones(1, 1, size, size, device=device)
    image = torch.cat([sar, opt], dim=1)
    return {"sar": sar, "opt": opt, "opt_mask": opt_mask, "image": image}


def _profile_macs(model: nn.Module, model_name: str, config: dict[str, Any], size: int, device: torch.device) -> tuple[int, tuple[int, ...]]:
    macs = {"value": 0}
    handles = []

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        if not torch.is_tensor(output):
            return
        out_elements = output.numel()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs["value"] += int(out_elements * kernel_ops)

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        if not torch.is_tensor(output):
            return
        macs["value"] += int(output.numel() * module.in_features)

    def attention_hook(module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        f_sar = inputs[0]
        if not torch.is_tensor(f_sar):
            return
        b, _, h, w = f_sar.shape
        window = max(1, min(int(module.window_size), h, w))
        pad_h = (window - h % window) % window
        pad_w = (window - w % window) % window
        hp, wp = h + pad_h, w + pad_w
        nwin = (hp // window) * (wp // window)
        tokens = window * window
        head_dim = int(module.hidden_dim) // int(module.num_heads)
        # QK^T and attention-value matmuls. Conv projections are counted by Conv2d hooks.
        macs["value"] += int(2 * b * nwin * int(module.num_heads) * tokens * tokens * head_dim)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (MaskAwareCrossAttentionBlock, MaskAwareCrossAttentionBlockDeep)):
            handles.append(module.register_forward_hook(attention_hook))

    batch = _make_batch(model_name, size, device)
    with torch.no_grad():
        output = model(select_model_input(batch, model_name))
        logits = resolve_segmentation_logits(output, batch, model_inference_mode(config))
    for handle in handles:
        handle.remove()
    return macs["value"], tuple(logits.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/model_complexity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for expected_model, label, config_path, notes in MODEL_SPECS:
        start = time.time()
        config = load_config(config_path)
        model_name = resolve_model_name(config, expected_model)
        model = build_model(model_name, config).to(device).eval()
        params, trainable = _count_params(model)
        macs, output_shape = _profile_macs(model, model_name, config, args.input_size, device)
        rows.append(
            {
                "model_id": model_name,
                "model": label,
                "config": config_path,
                "input_size": f"1x{args.input_size}x{args.input_size}",
                "output_shape": str(output_shape),
                "params": params,
                "params_M": params / 1_000_000,
                "trainable_params": trainable,
                "trainable_params_M": trainable / 1_000_000,
                "MACs": macs,
                "MACs_G": macs / 1_000_000_000,
                "FLOPs_2xMACs": macs * 2,
                "FLOPs_G": macs * 2 / 1_000_000_000,
                "notes": notes,
                "seconds_to_count": time.time() - start,
            }
        )
        print(json.dumps(rows[-1], indent=2))

    out_dir = ensure_dir(args.output_dir)
    csv_path = out_dir / f"model_complexity_{args.input_size}_all.csv"
    md_path = out_dir / f"model_complexity_{args.input_size}_all.md"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# Model Complexity at {args.input_size}x{args.input_size}\n\n"]
    lines.append("MACs count Conv2d/Linear operations plus explicit QK and AV matmuls in mask-aware cross-attention. FLOPs are reported as 2 x MACs.\n\n")
    lines.append("| model | Params (M) | MACs (G) | FLOPs (G) | output | notes |\n")
    lines.append("|---|---:|---:|---:|---|---|\n")
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['params_M']:.3f} | {row['MACs_G']:.3f} | {row['FLOPs_G']:.3f} | {row['output_shape']} | {row['notes']} |\n"
        )
    md_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
