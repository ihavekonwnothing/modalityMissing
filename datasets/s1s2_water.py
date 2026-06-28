from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


S1S2_BAND_ORDER = ["VV", "VH", "Blue", "Green", "Red", "NIR"]


@dataclass(frozen=True)
class S1S2WaterScene:
    sample_id: str
    split: str
    root: Path
    meta_path: Path
    assets: Dict[str, Path]
    metadata: Dict[str, Any]


def _require_rasterio():
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window, from_bounds
    except ImportError as exc:
        raise ImportError("S1S2WaterDataset requires rasterio for GeoTIFF window reads.") from exc
    return rasterio, Resampling, Window, from_bounds


def resolve_s1s2_root(root: Optional[str | Path] = None, root_env: str = "S1S2_WATER_ROOT") -> Path:
    value = root or os.environ.get(root_env)
    if not value:
        raise ValueError(f"Set {root_env} or pass dataset root explicitly.")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def discover_s1s2_water_scenes(root: str | Path) -> List[S1S2WaterScene]:
    root = Path(root).expanduser().resolve()
    scenes: List[S1S2WaterScene] = []
    for meta_path in sorted(root.glob("*/sentinel12_*_meta.json"), key=lambda p: int(p.parent.name)):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        props = meta.get("properties", {})
        assets: Dict[str, Path] = {}
        for key, asset in meta.get("assets", {}).items():
            href = Path(asset.get("href", ""))
            candidates = [root / href, meta_path.parent / href.name, meta_path.parent / href]
            for candidate in candidates:
                if candidate.exists():
                    assets[key] = candidate.resolve()
                    break
        required = {"s1_img", "s2_img", "s2_msk"}
        if not required.issubset(assets):
            missing = ", ".join(sorted(required - set(assets)))
            raise FileNotFoundError(f"Scene {meta_path.parent.name} missing required assets: {missing}")
        scenes.append(
            S1S2WaterScene(
                sample_id=meta_path.parent.name,
                split=props.get("split", "unknown"),
                root=root,
                meta_path=meta_path.resolve(),
                assets=assets,
                metadata=meta,
            )
        )
    if not scenes:
        raise FileNotFoundError(f"No S1S2-Water STAC metadata found under {root}")
    return scenes


def split_scenes(scenes: Sequence[S1S2WaterScene], split: str) -> List[S1S2WaterScene]:
    selected = [scene for scene in scenes if scene.split == split]
    if not selected:
        raise ValueError(f"No scenes found for split={split!r}")
    return selected


def _scene_s2_shape(scene: S1S2WaterScene) -> Tuple[int, int]:
    rasterio, _, _, _ = _require_rasterio()
    with rasterio.open(scene.assets["s2_img"]) as src:
        return int(src.height), int(src.width)


def _tile_grid(height: int, width: int, patch_size: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    stride = stride or patch_size
    ys = list(range(0, max(1, height - patch_size + 1), stride))
    xs = list(range(0, max(1, width - patch_size + 1), stride))
    if ys[-1] != max(0, height - patch_size):
        ys.append(max(0, height - patch_size))
    if xs[-1] != max(0, width - patch_size):
        xs.append(max(0, width - patch_size))
    return [(y, x) for y in ys for x in xs]


class S1S2WaterDataset:
    """S1S2-Water dataset returning 6-band water segmentation patches.

    Output channel order is [VV, VH, Blue, Green, Red, NIR]. DEM/elevation/slope
    assets are discovered but excluded from the main experiment.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        patch_size: int = 512,
        stats: Optional[Dict[str, Sequence[float]]] = None,
        training: Optional[bool] = None,
        patches_per_scene: int = 32,
        stride: Optional[int] = None,
        train_sampling: str = "random_crop",
        seed: int = 4,
        exclude_scenes: Optional[Sequence[str | int]] = None,
    ) -> None:
        self.root = resolve_s1s2_root(root)
        self.split = split
        self.patch_size = int(patch_size)
        self.training = (split == "train") if training is None else training
        self.patches_per_scene = int(patches_per_scene)
        self.train_sampling = train_sampling
        self.seed = int(seed)
        self.scenes = split_scenes(discover_s1s2_water_scenes(self.root), split)
        excluded = {str(scene_id) for scene_id in (exclude_scenes or [])}
        if excluded:
            self.scenes = [scene for scene in self.scenes if scene.sample_id not in excluded]
            if not self.scenes:
                raise ValueError(f"No scenes left for split={split!r} after excluding {sorted(excluded)}")
        self.stats = stats
        self._rng = np.random.default_rng(seed)
        self.tiles: List[Tuple[int, int, int]] = []
        if self.training and self.train_sampling not in {"random_crop", "tiles"}:
            raise ValueError("train_sampling must be 'random_crop' or 'tiles'")
        if (not self.training) or self.train_sampling == "tiles":
            for scene_idx, scene in enumerate(self.scenes):
                height, width = _scene_s2_shape(scene)
                for y, x in _tile_grid(height, width, self.patch_size, stride):
                    self.tiles.append((scene_idx, y, x))

    def __len__(self) -> int:
        if self.training and self.train_sampling == "random_crop":
            return max(1, len(self.scenes) * self.patches_per_scene)
        return len(self.tiles)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sampling = self.train_sampling if self.training else "tiles"
        if self.training and self.train_sampling == "random_crop":
            scene_idx = index % len(self.scenes)
            scene = self.scenes[scene_idx]
            height, width = _scene_s2_shape(scene)
            max_y = max(0, height - self.patch_size)
            max_x = max(0, width - self.patch_size)
            y = int(self._rng.integers(0, max_y + 1)) if max_y else 0
            x = int(self._rng.integers(0, max_x + 1)) if max_x else 0
        else:
            scene_idx, y, x = self.tiles[index]
            scene = self.scenes[scene_idx]
        sample = self.read_patch(scene, x=x, y=y, size=self.patch_size)
        sample["metadata"]["tile"] = {"x": x, "y": y, "size": self.patch_size, "split": self.split, "sampling": sampling}
        return sample

    def tile_counts_by_scene(self) -> Dict[str, int]:
        counts = {scene.sample_id: 0 for scene in self.scenes}
        for scene_idx, _, _ in self.tiles:
            counts[self.scenes[scene_idx].sample_id] += 1
        return counts

    def read_patch(self, scene: S1S2WaterScene, x: int, y: int, size: int) -> Dict[str, Any]:
        rasterio, Resampling, Window, from_bounds = _require_rasterio()
        s2_window = Window(x, y, size, size)
        with rasterio.open(scene.assets["s2_img"]) as s2_src:
            opt = s2_src.read(indexes=[1, 2, 3, 4], window=s2_window, boundless=True, fill_value=0).astype(np.float32)
            s2_transform = s2_src.window_transform(s2_window)
            bounds = rasterio.windows.bounds(s2_window, s2_src.transform)
            out_shape = (size, size)
        with rasterio.open(scene.assets["s1_img"]) as s1_src:
            s1_window = from_bounds(*bounds, transform=s1_src.transform)
            sar = s1_src.read(
                indexes=[1, 2],
                window=s1_window,
                out_shape=(2, *out_shape),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            ).astype(np.float32)
        with rasterio.open(scene.assets["s2_msk"]) as mask_src:
            mask = mask_src.read(1, window=s2_window, boundless=True, fill_value=0).astype(np.float32)[None]
        valid = np.ones_like(mask, dtype=np.float32)
        for key in ("s1_valid", "s2_valid"):
            if key in scene.assets:
                with rasterio.open(scene.assets[key]) as valid_src:
                    if key == "s1_valid":
                        valid_window = from_bounds(*bounds, transform=valid_src.transform)
                        arr = valid_src.read(
                            1,
                            window=valid_window,
                            out_shape=out_shape,
                            resampling=Resampling.nearest,
                            boundless=True,
                            fill_value=0,
                        )
                    else:
                        arr = valid_src.read(1, window=s2_window, boundless=True, fill_value=0)
                valid *= (arr > 0).astype(np.float32)[None]
        mask = (mask > 0).astype(np.float32)
        image = np.concatenate([sar, opt], axis=0)
        if self.stats:
            mean = np.asarray(self.stats["mean"], dtype=np.float32)[:, None, None]
            std = np.asarray(self.stats["std"], dtype=np.float32)[:, None, None]
            image = (image - mean) / np.maximum(std, 1e-6)
            image[2:] = np.where(np.isfinite(image[2:]), image[2:], 0.0)
            sar, opt = image[:2], image[2:]

        try:
            import torch

            to_tensor = torch.from_numpy
            image_t = to_tensor(image.astype(np.float32))
            sar_t = to_tensor(sar.astype(np.float32))
            opt_t = to_tensor(opt.astype(np.float32))
            mask_t = to_tensor(mask.astype(np.float32))
            valid_t = to_tensor(valid.astype(np.float32))
        except ImportError:
            image_t, sar_t, opt_t, mask_t, valid_t = image, sar, opt, mask, valid

        metadata = {
            "scene_metadata": scene.metadata.get("properties", {}),
            "assets": {k: str(v) for k, v in scene.assets.items()},
            "band_order": S1S2_BAND_ORDER,
            "s2_transform": tuple(s2_transform)[:6],
        }
        return {
            "image": image_t,
            "sar": sar_t,
            "opt": opt_t,
            "mask": mask_t,
            "valid_mask": valid_t,
            "sample_id": scene.sample_id,
            "metadata": metadata,
        }

    def scene_tile_count(self, sample_id: str) -> int:
        return sum(1 for scene_idx, _, _ in self.tiles if self.scenes[scene_idx].sample_id == sample_id)


def estimate_train_stats(
    root: str | Path,
    patch_size: int = 512,
    samples_per_scene: int = 4,
    seed: int = 4,
) -> Dict[str, List[float]]:
    dataset = S1S2WaterDataset(root=root, split="train", patch_size=patch_size, stats=None, patches_per_scene=1, seed=seed)
    rng = np.random.default_rng(seed)
    sums = np.zeros(6, dtype=np.float64)
    sums_sq = np.zeros(6, dtype=np.float64)
    counts = np.zeros(6, dtype=np.float64)
    for scene in dataset.scenes:
        height, width = _scene_s2_shape(scene)
        for _ in range(samples_per_scene):
            y = int(rng.integers(0, max(1, height - patch_size + 1)))
            x = int(rng.integers(0, max(1, width - patch_size + 1)))
            sample = dataset.read_patch(scene, x=x, y=y, size=patch_size)
            image = sample["image"].numpy() if hasattr(sample["image"], "numpy") else sample["image"]
            valid = sample["valid_mask"].numpy() if hasattr(sample["valid_mask"], "numpy") else sample["valid_mask"]
            valid2 = valid[0] > 0
            for c in range(6):
                values = image[c][valid2]
                if values.size:
                    sums[c] += float(values.sum())
                    sums_sq[c] += float((values * values).sum())
                    counts[c] += values.size
    mean = sums / np.maximum(counts, 1.0)
    var = sums_sq / np.maximum(counts, 1.0) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-6))
    return {"band_order": S1S2_BAND_ORDER, "mean": mean.tolist(), "std": std.tolist()}


def build_s1s2_water_datasets(config: Dict[str, Any], stats: Optional[Dict[str, Any]] = None):
    dataset_cfg = config["dataset"]
    root = resolve_s1s2_root(root_env=dataset_cfg.get("root_env", "S1S2_WATER_ROOT"))
    patch_size = int(dataset_cfg.get("patch_size", 512))
    train = S1S2WaterDataset(root, "train", patch_size=patch_size, stats=stats, training=True)
    val = S1S2WaterDataset(root, "val", patch_size=patch_size, stats=stats, training=False)
    test = S1S2WaterDataset(root, "test", patch_size=patch_size, stats=stats, training=False)
    return train, val, test
