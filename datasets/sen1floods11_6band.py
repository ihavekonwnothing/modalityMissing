from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


SEN1FLOODS11_BAND_ORDER = ["VV", "VH", "Blue", "Green", "Red", "NIR"]


def _require_rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("Sen1Floods116BandDataset requires rasterio. Activate the project conda environment first.") from exc
    return rasterio


def _load_source_stats(stats_path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(stats_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    stats = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(stats.get("mean"), dtype=np.float32)
    std = np.asarray(stats.get("std"), dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(f"Expected 6-channel mean/std in {path}, got mean={mean.shape}, std={std.shape}")
    order = stats.get("band_order")
    if order and list(order) != SEN1FLOODS11_BAND_ORDER:
        raise ValueError(f"Stats band order {order} does not match {SEN1FLOODS11_BAND_ORDER}")
    return {"mean": mean[:, None, None], "std": np.maximum(std[:, None, None], 1e-6)}


class Sen1Floods116BandDataset(Dataset):
    """Sen1Floods11 HandLabeled 6-band test-only dataset.

    Channel order is [VV, VH, Blue, Green, Red, NIR]. Normalization uses the
    source S1S2-Water train statistics passed with `stats_path`.
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        normalize: bool = True,
        stats_path: Optional[str | Path] = None,
        return_paths: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.normalize = bool(normalize)
        self.return_paths = bool(return_paths)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.metadata_path = self.root / "metadata.csv"
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        if self.normalize and stats_path is None:
            raise ValueError("stats_path is required when normalize=True")
        self.stats = _load_source_stats(stats_path) if self.normalize else None
        self.rows = self._read_metadata()
        if not self.rows:
            raise ValueError(f"No Sen1Floods11 rows found in {self.metadata_path}")

    def _read_metadata(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with self.metadata_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"chip_id", "image", "label", "valid_mask"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.metadata_path} missing columns: {sorted(missing)}")
            for row in reader:
                chip_id = row["chip_id"]
                image_path = self.root / row["image"]
                label_path = self.root / row["label"]
                valid_path = self.root / row["valid_mask"]
                for path in (image_path, label_path, valid_path):
                    if not path.exists():
                        raise FileNotFoundError(path)
                rows.append(
                    {
                        "chip_id": chip_id,
                        "image": str(image_path),
                        "label": str(label_path),
                        "valid_mask": str(valid_path),
                    }
                )
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rasterio = _require_rasterio()
        row = self.rows[index]
        with rasterio.open(row["image"]) as src:
            image = src.read().astype(np.float32)
            height, width = src.height, src.width
        if image.shape[0] != 6:
            raise ValueError(f"{row['image']} has {image.shape[0]} bands; expected 6")

        with rasterio.open(row["label"]) as src:
            label = src.read(1).astype(np.uint8)
        with rasterio.open(row["valid_mask"]) as src:
            valid_mask = src.read(1).astype(np.uint8)

        if label.shape != (height, width):
            raise ValueError(f"{row['label']} shape {label.shape} does not match image {(height, width)}")
        if valid_mask.shape != (height, width):
            raise ValueError(f"{row['valid_mask']} shape {valid_mask.shape} does not match image {(height, width)}")

        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        if self.stats is not None:
            image = (image - self.stats["mean"]) / self.stats["std"]
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        sample: Dict[str, Any] = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "sar": torch.from_numpy(image[:2].astype(np.float32)),
            "opt": torch.from_numpy(image[2:].astype(np.float32)),
            "label": torch.from_numpy((label > 0).astype(np.float32)[None]),
            "valid_mask": torch.from_numpy((valid_mask > 0).astype(np.float32)[None]),
            "chip_id": row["chip_id"],
        }
        if self.return_paths:
            sample["image_path"] = row["image"]
            sample["label_path"] = row["label"]
            sample["valid_mask_path"] = row["valid_mask"]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
