from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from models.factory import build_model, select_model_input
from models.output_utils import model_inference_mode, resolve_segmentation_logits
from tools.profile_model_complexity import MODEL_SPECS
from train import resolve_model_name
from utils.config import ensure_dir, load_config


def _make_batch(batch_size: int, size: int, device: torch.device) -> dict[str, torch.Tensor]:
    sar = torch.randn(batch_size, 2, size, size, device=device)
    opt = torch.randn(batch_size, 4, size, size, device=device)
    opt_mask = torch.ones(batch_size, 1, size, size, device=device)
    image = torch.cat([sar, opt], dim=1)
    return {"sar": sar, "opt": opt, "opt_mask": opt_mask, "image": image}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def _profile_one(
    model_name: str,
    label: str,
    config_path: str,
    notes: str,
    batch_size: int,
    input_size: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    config = load_config(config_path)
    resolved_name = resolve_model_name(config, model_name)
    model = build_model(resolved_name, config).to(device).eval()
    batch = _make_batch(batch_size, input_size, device)

    def forward() -> torch.Tensor:
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp and device.type == "cuda"):
            output = model(select_model_input(batch, resolved_name))
            return resolve_segmentation_logits(output, batch, model_inference_mode(config))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for _ in range(warmup):
        logits = forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times_ms: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits = forward()
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        import time

        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0
        for _ in range(repeats):
            start = time.perf_counter()
            logits = forward()
            times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = mean(times_ms)
    median_ms = median(times_ms)
    p95_ms = _percentile(times_ms, 0.95)
    return {
        "model_id": resolved_name,
        "model": label,
        "config": config_path,
        "batch_size": batch_size,
        "input_size": f"{batch_size}x{input_size}x{input_size}",
        "output_shape": str(tuple(logits.shape)),
        "amp": amp,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "warmup": warmup,
        "repeats": repeats,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95_ms,
        "fps_images_per_s": batch_size * 1000.0 / mean_ms,
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/model_complexity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model_id filter.")
    args = parser.parse_args()

    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    specs = MODEL_SPECS
    if args.models:
        wanted = set(args.models)
        specs = [spec for spec in MODEL_SPECS if spec[0] in wanted]
    rows = []
    for model_name, label, config_path, notes in specs:
        row = _profile_one(
            model_name=model_name,
            label=label,
            config_path=config_path,
            notes=notes,
            batch_size=args.batch_size,
            input_size=args.input_size,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
            amp=args.amp,
            amp_dtype=amp_dtype,
        )
        rows.append(row)
        print(json.dumps(row, indent=2))

    out_dir = ensure_dir(args.output_dir)
    suffix = f"{args.input_size}_bs{args.batch_size}_{'amp_' + args.amp_dtype if args.amp else 'fp32'}"
    csv_path = out_dir / f"inference_runtime_{suffix}.csv"
    md_path = out_dir / f"inference_runtime_{suffix}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# Inference Runtime at {args.input_size}x{args.input_size}, batch size {args.batch_size}\n\n"]
    lines.append(f"Device: `{rows[0]['device']}`. AMP: `{args.amp}`. dtype: `{args.amp_dtype if args.amp else 'fp32'}`. Warmup: {args.warmup}. Repeats: {args.repeats}.\n\n")
    lines.append("| model | Mean ms/img | Median ms/img | P95 ms/img | FPS | Peak allocated MB | Peak reserved MB |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        mean_per_img = row["mean_latency_ms"] / row["batch_size"]
        median_per_img = row["median_latency_ms"] / row["batch_size"]
        p95_per_img = row["p95_latency_ms"] / row["batch_size"]
        lines.append(
            f"| {row['model']} | {mean_per_img:.3f} | {median_per_img:.3f} | {p95_per_img:.3f} | {row['fps_images_per_s']:.2f} | {row['peak_allocated_mb']:.1f} | {row['peak_reserved_mb']:.1f} |\n"
        )
    md_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
