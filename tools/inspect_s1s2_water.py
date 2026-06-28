from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.s1s2_water import _tile_grid, discover_s1s2_water_scenes, resolve_s1s2_root


def _asset_info(path: Path) -> Dict[str, Any]:
    try:
        import rasterio

        with rasterio.open(path) as src:
            return {
                "shape": [src.count, src.height, src.width],
                "crs": str(src.crs),
                "resolution": [src.res[0], src.res[1]],
                "nodata": src.nodata,
                "dtype": src.dtypes[0] if src.dtypes else None,
            }
    except ImportError:
        return {"path": str(path), "note": "Install rasterio for shape/crs/resolution/nodata inspection."}


def inspect(root: str | None = None, max_samples: int = 5, patch_size: int = 512, stride: int | None = None) -> Dict[str, Any]:
    dataset_root = resolve_s1s2_root(root)
    scenes = discover_s1s2_water_scenes(dataset_root)
    split_counts = Counter(scene.split for scene in scenes)
    tile_counts = Counter()
    tile_counts_by_scene = {}
    sample_summaries = []
    for scene in scenes:
        s2_info = _asset_info(scene.assets["s2_img"])
        _, height, width = s2_info["shape"]
        n_tiles = len(_tile_grid(height, width, patch_size, stride))
        tile_counts[scene.split] += n_tiles
        tile_counts_by_scene[scene.sample_id] = n_tiles
    for scene in scenes[:max_samples]:
        props = scene.metadata.get("properties", {})
        s2_info = _asset_info(scene.assets["s2_img"])
        _, height, width = s2_info["shape"]
        sample_summaries.append(
            {
                "sample_id": scene.sample_id,
                "split": scene.split,
                "patch_size": patch_size,
                "stride": stride or patch_size,
                "num_tiles": len(_tile_grid(height, width, patch_size, stride)),
                "available_asset_keys": sorted(scene.assets),
                "assets": {key: _asset_info(path) for key, path in scene.assets.items()},
                "valid_mask_available": "s1_valid" in scene.assets and "s2_valid" in scene.assets,
                "date_s1": props.get("date_s1"),
                "date_s2": props.get("date_s2"),
                "date_difference_metadata": {
                    "date_s1": props.get("date_s1"),
                    "date_s2": props.get("date_s2"),
                },
            }
        )
    return {
        "root": str(dataset_root),
        "number_of_scenes": dict(split_counts),
        "patch_size": patch_size,
        "stride": stride or patch_size,
        "number_of_tiles": dict(tile_counts),
        "number_of_tiles_total": int(sum(tile_counts.values())),
        "tile_counts_by_scene": tile_counts_by_scene,
        "sample_ids": {split: [s.sample_id for s in scenes if s.split == split][:max_samples] for split in sorted(split_counts)},
        "samples": sample_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("S1S2_WATER_ROOT"))
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = inspect(args.root, args.max_samples, args.patch_size, args.stride)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
