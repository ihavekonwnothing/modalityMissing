from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.s1s2_water import S1S2WaterDataset, resolve_s1s2_root
from evaluate import resolve_model_name
from models.factory import build_model, select_model_input
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config
from utils.metrics import BinaryConfusion


@torch.no_grad()
def evaluate_excluding_scenes(
    config_path: str,
    checkpoint_path: str,
    exclude_scenes: set[str],
    split: str,
    output_csv: str,
    label: str | None = None,
    model_name: str | None = None,
) -> dict[str, float | str | int]:
    config = load_config(config_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    resolved_model_name = resolve_model_name(config, model_name, ckpt)
    output_dir = ensure_dir(config.get("output_dir", "outputs/s1s2_water/6band_main"))
    stats = ckpt.get("stats") or json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    root = resolve_s1s2_root(root_env=config["dataset"].get("root_env", "S1S2_WATER_ROOT"))
    ds = S1S2WaterDataset(
        root,
        split,
        patch_size=int(config["dataset"].get("patch_size", 256)),
        stats=stats,
        training=False,
    )
    original_tiles = len(ds.tiles)
    ds.tiles = [
        tile
        for tile in ds.tiles
        if ds.scenes[tile[0]].sample_id not in exclude_scenes
    ]
    kept_tiles = len(ds.tiles)
    loader = DataLoader(
        ds,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=segmentation_collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(resolved_model_name, config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    overall = BinaryConfusion()
    for batch in loader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = model(select_model_input(batch, resolved_model_name))
        overall.update(logits, batch["mask"], batch["valid_mask"])

    metrics = overall.compute()
    row = {
        "label": label or Path(checkpoint_path).stem,
        "model": resolved_model_name,
        "split": split,
        "excluded_scenes": ",".join(sorted(exclude_scenes)),
        "original_tiles": original_tiles,
        "kept_tiles": kept_tiles,
        **metrics,
    }
    out = Path(output_csv)
    ensure_dir(out.parent)
    write_header = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "model",
                "split",
                "excluded_scenes",
                "original_tiles",
                "kept_tiles",
                "IoU",
                "F1",
                "Precision",
                "Recall",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exclude-scene", action="append", default=[])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    row = evaluate_excluding_scenes(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        exclude_scenes=set(args.exclude_scene),
        split=args.split,
        output_csv=args.output_csv,
        label=args.label,
        model_name=args.model,
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
