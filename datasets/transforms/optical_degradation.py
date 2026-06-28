"""Controlled optical degradation / missing-modality simulation.

These transforms only modify Sentinel-2 channels [Blue, Green, Red, NIR].
They are not real cloud masks; they simulate missing or degraded optical input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DegradationResult:
    optical: np.ndarray
    missing_mask: np.ndarray


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _target_pixels(height: int, width: int, mask_ratio: float) -> int:
    return int(round(height * width * float(np.clip(mask_ratio, 0.0, 1.0))))


def random_block_mask(
    optical: np.ndarray,
    mask_ratio: float,
    min_blocks: int = 3,
    max_blocks: int = 12,
    seed: Optional[int] = None,
) -> DegradationResult:
    """Mask random rectangular regions in a C,H,W optical tensor."""
    if mask_ratio <= 0:
        return DegradationResult(optical.copy(), np.zeros(optical.shape[-2:], dtype=bool))
    if mask_ratio >= 1:
        return full_optical_missing(optical)

    out = optical.copy()
    h, w = out.shape[-2:]
    target = _target_pixels(h, w, mask_ratio)
    gen = _rng(seed)
    mask = np.zeros((h, w), dtype=bool)
    n_blocks = int(gen.integers(min_blocks, max_blocks + 1))
    attempts = 0
    while mask.sum() < target and attempts < n_blocks * 20:
        attempts += 1
        block_area = max(1, int((target - mask.sum()) / max(1, n_blocks)))
        aspect = float(gen.uniform(0.4, 2.5))
        bh = int(np.clip(np.sqrt(block_area / aspect), 8, h))
        bw = int(np.clip(bh * aspect, 8, w))
        y0 = int(gen.integers(0, max(1, h - bh + 1)))
        x0 = int(gen.integers(0, max(1, w - bw + 1)))
        mask[y0 : y0 + bh, x0 : x0 + bw] = True

    if mask.sum() > target:
        ys, xs = np.where(mask)
        keep = gen.choice(len(ys), size=target, replace=False)
        trimmed = np.zeros_like(mask)
        trimmed[ys[keep], xs[keep]] = True
        mask = trimmed
    out[:, mask] = 0.0
    return DegradationResult(out, mask)


def _box_blur(field: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return field
    padded = np.pad(field, radius, mode="reflect")
    out = np.zeros_like(field, dtype=np.float32)
    kernel = 2 * radius + 1
    for dy in range(kernel):
        for dx in range(kernel):
            out += padded[dy : dy + field.shape[0], dx : dx + field.shape[1]]
    return out / float(kernel * kernel)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="edge")
    out = np.zeros_like(mask, dtype=bool)
    kernel = 2 * radius + 1
    for dy in range(kernel):
        for dx in range(kernel):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="edge")
    out = np.ones_like(mask, dtype=bool)
    kernel = 2 * radius + 1
    for dy in range(kernel):
        for dx in range(kernel):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def cloud_like_mask(
    optical: np.ndarray,
    mask_ratio: float,
    seed: Optional[int] = None,
    blur_radius: Optional[int] = None,
    morphology_radius: int = 2,
) -> DegradationResult:
    """Mask continuous irregular regions using smoothed random fields."""
    if mask_ratio <= 0:
        return DegradationResult(optical.copy(), np.zeros(optical.shape[-2:], dtype=bool))
    if mask_ratio >= 1:
        return full_optical_missing(optical)

    out = optical.copy()
    h, w = out.shape[-2:]
    gen = _rng(seed)
    field = gen.normal(size=(h, w)).astype(np.float32)
    if blur_radius is None:
        blur_radius = max(3, min(h, w) // 32)
    for radius in (blur_radius, max(1, blur_radius // 2)):
        field = _box_blur(field, radius)
    threshold = np.quantile(field, 1.0 - float(mask_ratio))
    mask = field >= threshold
    mask = _binary_dilate(mask, morphology_radius)
    mask = _binary_erode(mask, max(1, morphology_radius // 2))

    target = _target_pixels(h, w, mask_ratio)
    ys, xs = np.where(mask)
    if len(ys) > target:
        order = np.argsort(field[ys, xs])[-target:]
        trimmed = np.zeros_like(mask)
        trimmed[ys[order], xs[order]] = True
        mask = trimmed
    out[:, mask] = 0.0
    return DegradationResult(out, mask)


def full_optical_missing(optical: np.ndarray) -> DegradationResult:
    out = np.zeros_like(optical, dtype=np.float32)
    return DegradationResult(out, np.ones(optical.shape[-2:], dtype=bool))


def apply_optical_degradation(
    optical: np.ndarray,
    mask_type: str,
    mask_ratio: float = 0.0,
    seed: Optional[int] = None,
) -> DegradationResult:
    if mask_type in ("none", "clean") or mask_ratio <= 0:
        return DegradationResult(optical.copy(), np.zeros(optical.shape[-2:], dtype=bool))
    if mask_type == "random_block_mask":
        return random_block_mask(optical, mask_ratio, seed=seed)
    if mask_type == "cloud_like_mask":
        return cloud_like_mask(optical, mask_ratio, seed=seed)
    if mask_type == "full_optical_missing" or mask_ratio >= 1:
        return full_optical_missing(optical)
    raise ValueError(f"Unknown optical degradation type: {mask_type}")
