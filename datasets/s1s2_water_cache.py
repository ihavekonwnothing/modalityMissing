from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

CACHE_BUILDING_FILE = "CACHE_BUILDING"
CACHE_COMPLETE_FILE = "CACHE_COMPLETE"


def is_patch_cache_ready(cache_dir: str | Path) -> bool:
    cache_dir = Path(cache_dir)
    return (
        (cache_dir / "manifest.csv").exists()
        and (cache_dir / "metadata.json").exists()
        and (cache_dir / CACHE_COMPLETE_FILE).exists()
        and not (cache_dir / CACHE_BUILDING_FILE).exists()
    )


class S1S2WaterPatchCacheDataset:
    """Dataset backed by precomputed S1S2-Water 512x512 patch files."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        exclude_scenes: Sequence[str | int] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        if not is_patch_cache_ready(self.cache_dir):
            raise FileNotFoundError(f"Patch cache is not complete: {self.cache_dir}")
        manifest_path = self.cache_dir / "manifest.csv"
        excluded = {str(scene_id) for scene_id in (exclude_scenes or [])}
        with manifest_path.open(newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row["split"] == split and row["sample_id"] not in excluded]
        if not rows:
            raise ValueError(f"No cached patches found for split={split!r} under {self.cache_dir}")
        self.rows = rows
        meta_path = self.cache_dir / "metadata.json"
        self.cache_metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        patch_path = self.cache_dir / row["path"]
        try:
            with np.load(patch_path) as data:
                sar = data["sar"].astype(np.float32)
                opt = data["opt"].astype(np.float32)
                mask = data["mask"].astype(np.float32)[None]
                valid = data["valid_mask"].astype(np.float32)[None]
        except Exception as exc:
            raise RuntimeError(f"Failed to load cached S1S2-Water patch: {patch_path}") from exc
        image = np.concatenate([sar, opt], axis=0)
        return {
            "image": torch.from_numpy(image),
            "sar": torch.from_numpy(sar),
            "opt": torch.from_numpy(opt),
            "mask": torch.from_numpy(mask),
            "valid_mask": torch.from_numpy(valid),
            "sample_id": row["sample_id"],
            "metadata": {
                "tile": {
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "size": int(row["size"]),
                    "split": row["split"],
                    "sampling": "cached_tiles",
                },
                "cache_path": str(self.cache_dir / row["path"]),
            },
        }
