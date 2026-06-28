from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from datasets.s1s2_water_cache import CACHE_BUILDING_FILE, CACHE_COMPLETE_FILE
from datasets.s1s2_water import S1S2WaterDataset, estimate_train_stats, resolve_s1s2_root
from utils.config import ensure_dir


def _load_or_create_stats(root: Path, output_dir: Path, patch_size: int, samples_per_scene: int, seed: int):
    stats_path = output_dir / "stats.json"
    if stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    stats = estimate_train_stats(root, patch_size=patch_size, samples_per_scene=samples_per_scene, seed=seed)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _to_numpy(value):
    return value.numpy() if hasattr(value, "numpy") else value


def _valid_patch_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            return all(key in data for key in ("sar", "opt", "mask", "valid_mask"))
    except Exception:
        return False


def _write_patch_file(path: Path, sar: np.ndarray, opt: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> None:
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, sar=sar, opt=opt, mask=mask, valid_mask=valid)
    tmp_path.replace(path)


def build_cache(args) -> None:
    root = resolve_s1s2_root(args.root, root_env=args.root_env)
    output_dir = ensure_dir(args.output_dir)
    building_marker = output_dir / CACHE_BUILDING_FILE
    complete_marker = output_dir / CACHE_COMPLETE_FILE
    if complete_marker.exists():
        complete_marker.unlink()
    building_marker.write_text("building\n", encoding="utf-8")
    stats = _load_or_create_stats(root, output_dir, args.patch_size, args.stats_samples_per_scene, args.seed)
    rows = []
    try:
        for split in args.splits:
            ds = S1S2WaterDataset(
                root,
                split,
                patch_size=args.patch_size,
                stats=stats,
                training=(split == "train"),
                train_sampling="tiles",
                stride=args.stride,
                seed=args.seed,
                exclude_scenes=args.exclude_scene,
            )
            tiles = ds.tiles[: args.max_patches_per_split] if args.max_patches_per_split else ds.tiles
            iterator = tqdm(tiles, desc=f"cache {split}", dynamic_ncols=True)
            for tile_index, (scene_idx, y, x) in enumerate(iterator):
                scene = ds.scenes[scene_idx]
                rel = Path(split) / scene.sample_id / f"y{y:05d}_x{x:05d}.npz"
                out_file = output_dir / rel
                if not args.overwrite and _valid_patch_file(out_file):
                    rows.append({"split": split, "sample_id": scene.sample_id, "x": x, "y": y, "size": args.patch_size, "path": str(rel)})
                    continue
                ensure_dir(out_file.parent)
                sample = ds.read_patch(scene, x=x, y=y, size=args.patch_size)
                sar = _to_numpy(sample["sar"]).astype(np.float16)
                opt = _to_numpy(sample["opt"]).astype(np.float16)
                mask = (_to_numpy(sample["mask"])[0] > 0.5).astype(np.uint8)
                valid = (_to_numpy(sample["valid_mask"])[0] > 0.5).astype(np.uint8)
                _write_patch_file(out_file, sar=sar, opt=opt, mask=mask, valid=valid)
                rows.append({"split": split, "sample_id": scene.sample_id, "x": x, "y": y, "size": args.patch_size, "path": str(rel)})
        manifest = output_dir / "manifest.csv"
        manifest_tmp = output_dir / "manifest.tmp"
        with manifest_tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "sample_id", "x", "y", "size", "path"])
            writer.writeheader()
            writer.writerows(rows)
        manifest_tmp.replace(manifest)
        metadata = {
            "dataset": "s1s2_water",
            "root": str(root),
            "patch_size": args.patch_size,
            "stride": args.stride,
            "splits": args.splits,
            "exclude_scene": [str(v) for v in args.exclude_scene],
            "storage": "npz_compressed_normalized_float16",
            "stats": stats,
            "num_patches": len(rows),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        complete_marker.write_text("complete\n", encoding="utf-8")
        building_marker.unlink(missing_ok=True)
        print(json.dumps({"cache_dir": str(output_dir), "num_patches": len(rows), "manifest": str(manifest)}, indent=2))
    except Exception:
        complete_marker.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--root-env", default="S1S2_WATER_ROOT")
    parser.add_argument("--output-dir", default="data/s1s2_water_patch_cache_512")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--exclude-scene", action="append", default=["31"])
    parser.add_argument("--stats-samples-per-scene", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-patches-per-split", type=int, default=None)
    args = parser.parse_args()
    build_cache(args)


if __name__ == "__main__":
    main()
