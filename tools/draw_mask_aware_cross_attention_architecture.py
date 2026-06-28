from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("outputs/figures/model_architecture")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, xy, w, h, text, fc="#f4f6f8", ec="#4a5568", fontsize=10, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#1f2937",
    )


def _arrow(ax, p1, p2, color="#4a5568", lw=1.4, style="-|>", mutation_scale=12):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
        )
    )


def draw_overview(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    _box(ax, (0.2, 4.4), 1.5, 0.7, "SAR\nVV, VH", fc="#dbeafe", ec="#2563eb", fontsize=10, weight="bold")
    _box(ax, (0.2, 2.9), 1.5, 0.7, "OPT\nB, G, R, NIR", fc="#dcfce7", ec="#16a34a", fontsize=10, weight="bold")
    _box(ax, (0.2, 1.4), 1.5, 0.7, "Opt mask\n1 valid / 0 missing", fc="#fef3c7", ec="#d97706", fontsize=9, weight="bold")

    _box(ax, (2.1, 4.2), 1.4, 0.9, "SAR encoder", fc="#eff6ff", ec="#2563eb", fontsize=10)
    _box(ax, (2.1, 2.7), 1.4, 0.9, "Opt encoder", fc="#ecfdf5", ec="#16a34a", fontsize=10)
    _box(ax, (2.1, 1.2), 1.4, 0.9, "Mask path", fc="#fffbeb", ec="#d97706", fontsize=10)

    _box(ax, (4.1, 4.15), 1.9, 1.0, "Stage 1\nMask-Guided Fusion", fc="#eef2ff", ec="#4338ca", fontsize=10, weight="bold")
    _box(ax, (4.1, 2.65), 1.9, 1.0, "Stage 2\nMask-Guided Fusion", fc="#eef2ff", ec="#4338ca", fontsize=10, weight="bold")
    _box(ax, (4.1, 1.15), 1.9, 1.0, "Stage 3\nMask-Aware Cross-Attention", fc="#f5f3ff", ec="#7c3aed", fontsize=10, weight="bold")

    _box(ax, (6.6, 2.9), 1.6, 1.1, "U-Net\ndecoder", fc="#f8fafc", ec="#64748b", fontsize=10, weight="bold")
    _box(ax, (8.7, 3.9), 0.95, 0.75, "Fused\nhead", fc="#e0f2fe", ec="#0ea5e9", fontsize=9, weight="bold")
    _box(ax, (8.7, 2.55), 0.95, 0.75, "SAR\nhead", fc="#fae8ff", ec="#c026d3", fontsize=9, weight="bold")
    _box(ax, (8.55, 1.0), 1.1, 0.9, "Adaptive\nfallback", fc="#fff1f2", ec="#e11d48", fontsize=9, weight="bold")

    # Connections
    _arrow(ax, (1.7, 4.75), (2.1, 4.75), color="#2563eb")
    _arrow(ax, (1.7, 3.25), (2.1, 3.25), color="#16a34a")
    _arrow(ax, (1.7, 1.75), (2.1, 1.75), color="#d97706")
    _arrow(ax, (3.5, 4.75), (4.1, 4.75), color="#2563eb")
    _arrow(ax, (3.5, 3.25), (4.1, 3.25), color="#16a34a")
    _arrow(ax, (3.5, 1.75), (4.1, 1.75), color="#d97706")
    _arrow(ax, (6.0, 4.65), (6.6, 4.25), color="#4338ca")
    _arrow(ax, (6.0, 3.15), (6.6, 3.25), color="#4338ca")
    _arrow(ax, (6.0, 1.65), (6.6, 3.15), color="#7c3aed")
    _arrow(ax, (8.2, 3.45), (8.7, 4.28), color="#0ea5e9")
    _arrow(ax, (8.2, 3.05), (8.7, 2.92), color="#c026d3")
    _arrow(ax, (9.15, 2.55), (9.15, 1.9), color="#e11d48")
    _arrow(ax, (9.15, 1.9), (8.85, 1.9), color="#e11d48")
    ax.text(5.0, 5.55, "Mask-Aware Cross-Attention Fusion U-Net", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(5.0, 0.35, "Dual encoder, late fusion, SAR auxiliary branch, adaptive fallback at inference", ha="center", va="center", fontsize=9, color="#374151")


def draw_fusion_block(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.55, "Mask-Guided Fusion", fontsize=14, weight="bold")
    _box(ax, (0.4, 3.9), 1.45, 0.75, "f_sar", fc="#dbeafe", ec="#2563eb", fontsize=11, weight="bold")
    _box(ax, (0.4, 1.95), 1.45, 0.75, "f_opt", fc="#dcfce7", ec="#16a34a", fontsize=11, weight="bold")
    _box(ax, (0.4, 0.45), 1.45, 0.75, "mask", fc="#fef3c7", ec="#d97706", fontsize=11, weight="bold")

    _box(ax, (2.4, 2.55), 1.35, 1.15, "mask\nresize", fc="#fff7ed", ec="#ea580c", fontsize=10, weight="bold")
    _box(ax, (4.05, 2.55), 1.35, 1.15, "opt_proj\n1x1 conv", fc="#ede9fe", ec="#7c3aed", fontsize=10, weight="bold")
    _box(ax, (5.75, 2.25), 1.95, 1.7, "gate\nsigmoid(Conv)", fc="#f8fafc", ec="#475569", fontsize=10, weight="bold")
    _box(ax, (8.2, 2.55), 1.25, 1.1, "f_fused", fc="#dbeafe", ec="#2563eb", fontsize=11, weight="bold")

    _arrow(ax, (1.85, 4.28), (2.4, 3.4), color="#2563eb")
    _arrow(ax, (1.85, 2.32), (2.4, 3.0), color="#16a34a")
    _arrow(ax, (1.85, 0.82), (2.4, 2.9), color="#d97706")
    _arrow(ax, (3.75, 3.12), (4.05, 3.12), color="#ea580c")
    _arrow(ax, (5.4, 3.12), (5.75, 3.12), color="#7c3aed")
    _arrow(ax, (7.7, 3.12), (8.2, 3.12), color="#2563eb")
    ax.text(5.0, 0.95, "f_fused = f_sar + gate * opt_proj", ha="center", va="center", fontsize=10, family="monospace")


def draw_attention_block(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.55, "Mask-Aware Cross-Attention", fontsize=14, weight="bold")
    _box(ax, (0.4, 4.15), 1.45, 0.75, "f_sar", fc="#dbeafe", ec="#2563eb", fontsize=11, weight="bold")
    _box(ax, (0.4, 2.05), 1.45, 0.75, "f_opt", fc="#dcfce7", ec="#16a34a", fontsize=11, weight="bold")
    _box(ax, (0.4, 0.55), 1.45, 0.75, "mask", fc="#fef3c7", ec="#d97706", fontsize=11, weight="bold")

    _box(ax, (2.35, 4.0), 1.25, 1.0, "Q\nproj_q", fc="#eff6ff", ec="#2563eb", fontsize=10, weight="bold")
    _box(ax, (2.35, 2.05), 1.25, 1.0, "K\nproj_k", fc="#ecfdf5", ec="#16a34a", fontsize=10, weight="bold")
    _box(ax, (2.35, 0.55), 1.25, 1.0, "V\nproj_v", fc="#ecfdf5", ec="#16a34a", fontsize=10, weight="bold")
    _box(ax, (4.25, 2.0), 1.6, 1.2, "windowed\nattention", fc="#f5f3ff", ec="#7c3aed", fontsize=10, weight="bold")
    _box(ax, (6.25, 2.0), 1.3, 1.2, "gamma", fc="#fff1f2", ec="#e11d48", fontsize=10, weight="bold")
    _box(ax, (8.2, 2.0), 1.2, 1.2, "f_out", fc="#dbeafe", ec="#2563eb", fontsize=11, weight="bold")

    _arrow(ax, (1.85, 4.52), (2.35, 4.52), color="#2563eb")
    _arrow(ax, (1.85, 2.42), (2.35, 2.42), color="#16a34a")
    _arrow(ax, (1.85, 0.92), (2.35, 1.02), color="#d97706")
    _arrow(ax, (3.6, 4.52), (4.25, 2.8), color="#2563eb")
    _arrow(ax, (3.6, 2.42), (4.25, 2.55), color="#16a34a")
    _arrow(ax, (3.6, 1.05), (4.25, 2.2), color="#16a34a")
    _arrow(ax, (5.85, 2.6), (6.25, 2.6), color="#7c3aed")
    _arrow(ax, (7.55, 2.6), (8.2, 2.6), color="#2563eb")
    ax.text(5.1, 0.88, "Q = proj_q(f_sar)\nK,V = proj_k/v(f_opt * mask)\nout = f_sar + gamma * attn(Q,K,V) * mask", ha="center", va="center", fontsize=8.5, family="monospace")


def draw_heads(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.55, "Segmentation Heads", fontsize=14, weight="bold")
    _box(ax, (0.5, 3.6), 2.0, 1.0, "shared decoder\nfeatures", fc="#f8fafc", ec="#64748b", fontsize=10, weight="bold")
    _box(ax, (3.25, 4.05), 1.7, 0.9, "fused head\n1x1 conv", fc="#e0f2fe", ec="#0ea5e9", fontsize=10, weight="bold")
    _box(ax, (3.25, 2.25), 1.7, 0.9, "SAR head\n1x1 conv", fc="#fae8ff", ec="#c026d3", fontsize=10, weight="bold")
    _box(ax, (6.0, 4.05), 1.8, 0.9, "logits_fused", fc="#dbeafe", ec="#2563eb", fontsize=11, weight="bold")
    _box(ax, (6.0, 2.25), 1.8, 0.9, "logits_sar", fc="#f3e8ff", ec="#a855f7", fontsize=11, weight="bold")
    _box(ax, (8.1, 2.95), 1.5, 1.1, "adaptive\nfallback", fc="#fff1f2", ec="#e11d48", fontsize=10, weight="bold")
    _box(ax, (8.1, 0.95), 1.5, 1.1, "final\nprediction", fc="#dcfce7", ec="#16a34a", fontsize=10, weight="bold")

    _arrow(ax, (2.5, 4.1), (3.25, 4.5), color="#64748b")
    _arrow(ax, (2.5, 4.0), (3.25, 2.7), color="#64748b")
    _arrow(ax, (4.95, 4.5), (6.0, 4.5), color="#0ea5e9")
    _arrow(ax, (4.95, 2.7), (6.0, 2.7), color="#c026d3")
    _arrow(ax, (7.8, 4.5), (8.1, 3.9), color="#2563eb")
    _arrow(ax, (7.8, 2.7), (8.1, 3.4), color="#a855f7")
    _arrow(ax, (8.85, 2.95), (8.85, 2.05), color="#e11d48")
    _arrow(ax, (8.85, 2.05), (8.85, 2.05), color="#e11d48")
    _arrow(ax, (8.85, 2.05), (8.85, 2.05), color="#e11d48")
    _arrow(ax, (8.85, 1.95), (8.85, 2.05), color="#e11d48")
    _arrow(ax, (8.85, 1.95), (8.85, 1.5), color="#16a34a")
    _arrow(ax, (9.6, 2.5), (9.6, 1.5), color="#e11d48")
    _arrow(ax, (9.6, 1.5), (8.85, 1.5), color="#e11d48")
    ax.text(5.0, 0.35, "loss = loss_fused + 0.3 * loss_sar", ha="center", va="center", fontsize=11, family="monospace")
    ax.text(8.9, 2.25, "mask=0 -> logits_sar\nmask=1 -> logits_fused", ha="center", va="center", fontsize=8)


def main():
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    draw_overview(ax0)
    ax1 = fig.add_subplot(gs[1, 0])
    draw_fusion_block(ax1)
    ax2 = fig.add_subplot(gs[1, 1])
    draw_attention_block(ax2)
    ax3 = fig.add_subplot(gs[1, 2])
    draw_heads(ax3)

    out_png = OUT_DIR / "mask_aware_cross_attention_fusion_unet_architecture.png"
    out_svg = OUT_DIR / "mask_aware_cross_attention_fusion_unet_architecture.svg"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
