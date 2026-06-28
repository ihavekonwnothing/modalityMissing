from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from datasets.s1s2_water import S1S2WaterDataset, estimate_train_stats, resolve_s1s2_root
from datasets.s1s2_water_cache import S1S2WaterPatchCacheDataset, is_patch_cache_ready
from losses import segmentation_loss
from models.factory import build_model, select_model_input
from models.output_utils import model_inference_mode, resolve_segmentation_logits, segmentation_output_loss
from utils.collate import segmentation_collate
from utils.config import ensure_dir, load_config
from utils.metrics import BinaryConfusion
from utils.seed import set_seed
from utils.training_logger import append_epoch_metrics, setup_file_logger, write_training_summary


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 10
    min_epochs: int = 0
    min_delta: float = 0.0
    monitor: str = "val_IoU"


@dataclass
class EarlyStoppingState:
    best_score: float = -1.0
    best_epoch: int = 0
    stale: int = 0


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed_from_env() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(enabled=False)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        try:
            torch.distributed.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            torch.distributed.init_process_group(backend=backend)
    else:
        torch.distributed.init_process_group(backend=backend)
    return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _null_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def build_early_stopping_config(config) -> EarlyStoppingConfig:
    training = config.get("training", {})
    raw = training.get("early_stopping")
    if isinstance(raw, dict):
        return EarlyStoppingConfig(
            enabled=bool(raw.get("enabled", True)),
            patience=int(raw.get("patience", training.get("early_stopping_patience", 10))),
            min_epochs=int(raw.get("min_epochs", 0)),
            min_delta=float(raw.get("min_delta", 0.0)),
            monitor=str(raw.get("monitor", "val_IoU")),
        )
    return EarlyStoppingConfig(
        enabled=bool(training.get("early_stopping_enabled", True)),
        patience=int(training.get("early_stopping_patience", 10)),
        min_epochs=int(training.get("early_stopping_min_epochs", 0)),
        min_delta=float(training.get("early_stopping_min_delta", 0.0)),
        monitor=str(training.get("early_stopping_monitor", "val_IoU")),
    )


def update_early_stopping(
    state: EarlyStoppingState,
    score: float,
    epoch: int,
    cfg: EarlyStoppingConfig,
) -> tuple[EarlyStoppingState, bool, bool]:
    improved = score > (state.best_score + cfg.min_delta)
    if improved:
        state.best_score = score
        state.best_epoch = epoch
        state.stale = 0
    else:
        state.stale += 1
    should_stop = cfg.enabled and epoch >= cfg.min_epochs and state.stale >= cfg.patience
    return state, improved, should_stop


def resolve_model_name(config, model_name: str | None = None) -> str:
    if model_name:
        return model_name
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict) and model_cfg.get("name"):
        return str(model_cfg["name"])
    raise ValueError("Model name must be provided with --model or config.model.name")


def _stats_path(output_dir: Path) -> Path:
    return output_dir / "stats.json"


def _cache_metadata(config) -> dict:
    cache_dir = config.get("dataset", {}).get("cache_dir")
    if not cache_dir or not is_patch_cache_ready(cache_dir):
        return {}
    metadata_path = Path(cache_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_training_root(config) -> Path | None:
    dataset_cfg = config.get("dataset", {})
    root_env = dataset_cfg.get("root_env", "S1S2_WATER_ROOT")
    try:
        return resolve_s1s2_root(root_env=root_env)
    except ValueError:
        metadata = _cache_metadata(config)
        root = metadata.get("root")
        if root:
            path = Path(root).expanduser().resolve()
            if path.exists():
                return path
        if dataset_cfg.get("cache_dir") and is_patch_cache_ready(dataset_cfg["cache_dir"]):
            return None
        raise


def load_or_create_stats(config, root: Path | None, output_dir: Path):
    stats_file = _stats_path(output_dir)
    legacy_stats_file = Path("outputs/s1s2_water/stats.json")
    if stats_file.exists():
        return json.loads(stats_file.read_text(encoding="utf-8"))
    metadata = _cache_metadata(config)
    cache_stats = metadata.get("stats")
    if cache_stats:
        ensure_dir(stats_file.parent)
        ensure_dir(legacy_stats_file.parent)
        stats_file.write_text(json.dumps(cache_stats, indent=2), encoding="utf-8")
        legacy_stats_file.write_text(json.dumps(cache_stats, indent=2), encoding="utf-8")
        return cache_stats
    if root is None:
        raise ValueError("Cannot estimate stats without S1S2-Water root. Set S1S2_WATER_ROOT or provide cache metadata stats.")
    stats = estimate_train_stats(
        root,
        patch_size=int(config["dataset"].get("patch_size", 256)),
        samples_per_scene=int(config["dataset"].get("stats_samples_per_scene", 2)),
        seed=int(config["training"].get("seed", 4)),
    )
    ensure_dir(stats_file.parent)
    ensure_dir(legacy_stats_file.parent)
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    legacy_stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _normalize_mask_type(mask_type: str) -> str:
    if mask_type == "random_block":
        return "random_block_mask"
    if mask_type == "cloud_like":
        return "cloud_like_mask"
    return mask_type


def _target_mask_pixels(height: int, width: int, ratio: float) -> int:
    return int(round(height * width * max(0.0, min(1.0, float(ratio)))))


def _random_block_missing_mask(height: int, width: int, ratio: float, device: torch.device) -> torch.Tensor:
    target = _target_mask_pixels(height, width, ratio)
    mask = torch.zeros((height, width), device=device, dtype=torch.bool)
    if target <= 0:
        return mask
    if target >= height * width:
        return torch.ones_like(mask)
    n_blocks = int(torch.randint(3, 13, (1,), device=device).item())
    attempts = 0
    while int(mask.sum().item()) < target and attempts < n_blocks * 20:
        attempts += 1
        remaining = max(1, target - int(mask.sum().item()))
        block_area = max(1, remaining // max(1, n_blocks))
        aspect = float((torch.rand(1, device=device) * 2.1 + 0.4).item())
        bh = int(max(8, min(height, round((block_area / aspect) ** 0.5))))
        bw = int(max(8, min(width, round(bh * aspect))))
        y0 = int(torch.randint(0, max(1, height - bh + 1), (1,), device=device).item())
        x0 = int(torch.randint(0, max(1, width - bw + 1), (1,), device=device).item())
        mask[y0 : y0 + bh, x0 : x0 + bw] = True
    current = int(mask.sum().item())
    if current > target:
        idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        keep = idx[torch.randperm(idx.numel(), device=device)[:target]]
        trimmed = torch.zeros_like(mask.flatten())
        trimmed[keep] = True
        mask = trimmed.view(height, width)
    return mask


def _cloud_like_missing_mask(height: int, width: int, ratio: float, device: torch.device) -> torch.Tensor:
    target = _target_mask_pixels(height, width, ratio)
    if target <= 0:
        return torch.zeros((height, width), device=device, dtype=torch.bool)
    if target >= height * width:
        return torch.ones((height, width), device=device, dtype=torch.bool)
    field = torch.randn((1, 1, height, width), device=device)
    kernel = max(7, (min(height, width) // 32) | 1)
    padding = kernel // 2
    field = torch.nn.functional.avg_pool2d(field, kernel_size=kernel, stride=1, padding=padding)
    kernel2 = max(3, kernel // 2)
    kernel2 = kernel2 | 1
    field = torch.nn.functional.avg_pool2d(field, kernel_size=kernel2, stride=1, padding=kernel2 // 2)
    flat = field.flatten()
    topk = torch.topk(flat, k=target, largest=True).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[topk] = True
    return mask.view(height, width)


def _torch_optical_missing(optical: torch.Tensor, mask_type: str, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = optical.shape[-2:]
    mask_type = _normalize_mask_type(mask_type)
    if ratio <= 0 or mask_type in {"none", "clean"}:
        missing = torch.zeros((height, width), device=optical.device, dtype=torch.bool)
    elif ratio >= 1 or mask_type == "full_optical_missing":
        missing = torch.ones((height, width), device=optical.device, dtype=torch.bool)
    elif mask_type == "random_block_mask":
        missing = _random_block_missing_mask(height, width, ratio, optical.device)
    elif mask_type == "cloud_like_mask":
        missing = _cloud_like_missing_mask(height, width, ratio, optical.device)
    else:
        raise ValueError(f"Unknown optical missing mask type: {mask_type}")
    out = optical.clone()
    out[:, missing] = 0.0
    availability = (~missing).to(dtype=optical.dtype)
    return out, availability


def _apply_controlled_optical_missing(batch, config):
    robust = config.get("robust_fusion", {})
    if not robust.get("enabled", False):
        batch["opt_mask"] = torch.ones_like(batch["opt"][:, :1])
        return batch
    p_clean = float(robust.get("p_clean", 0.4))
    p_partial = float(robust.get("p_optical_partial_missing", 0.4))
    p_full = float(robust.get("p_optical_full_missing", 0.2))
    ratios = robust.get("mask_ratios", [0.25, 0.5, 0.75])
    mask_types = robust.get("mask_types", ["random_block", "cloud_like"])
    images = batch["image"]
    opts = batch["opt"]
    opt_masks = torch.ones((opts.shape[0], 1, opts.shape[-2], opts.shape[-1]), device=opts.device, dtype=opts.dtype)
    for i in range(images.shape[0]):
        r = torch.rand(1).item()
        if r < p_clean:
            pass
        elif r < p_clean + p_full:
            opts[i].zero_()
            opt_masks[i].zero_()
        elif r < p_clean + p_full + p_partial:
            ratio = float(ratios[int(torch.randint(0, len(ratios), (1,)).item())])
            mask_type = str(mask_types[int(torch.randint(0, len(mask_types), (1,)).item())])
            opts[i], opt_masks[i, 0] = _torch_optical_missing(opts[i], mask_type, ratio)
        images[i, 2:] = opts[i]
    batch["opt_mask"] = opt_masks
    return batch


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    model_name: str,
    device: torch.device,
    config: dict,
    max_batches: int | None = None,
    progress: bool = True,
    epoch: int | None = None,
    distributed: bool = False,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
):
    model.eval()
    meter = BinaryConfusion()
    total_loss = 0.0
    n = 0
    total = min(len(loader), max_batches) if max_batches is not None else len(loader)
    iterator = tqdm(loader, total=total, desc=f"epoch {epoch} val" if epoch else "val", leave=False, dynamic_ncols=True, disable=not progress)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        if model_name in {
            "mask_guided_late_fusion_unet",
            "mask_guided_late_fusion_sar_aux_unet",
            "mask_aware_cross_attention_fusion_unet",
            "mask_aware_cross_attention_no_sar_aux_unet",
            "mask_aware_cross_attention_fusion_unet_deep",
            "smagnet",
        } and "opt_mask" not in batch:
            batch["opt_mask"] = torch.ones_like(batch["opt"][:, :1])
        with _autocast_context(amp_enabled, amp_dtype, device):
            output = model(select_model_input(batch, model_name))
            logits = resolve_segmentation_logits(output, batch, model_inference_mode(config))
        loss, loss_ok = _safe_segmentation_loss(output, batch["mask"], batch["valid_mask"], config) if not torch.is_tensor(output) else _safe_segmentation_loss(logits, batch["mask"], batch["valid_mask"], config)
        logits_ok = bool(torch.isfinite(logits).all().item())
        if logits_ok:
            meter.update(logits.float(), batch["mask"], batch["valid_mask"])
        if loss_ok:
            total_loss += float(loss.item())
            n += 1
        iterator.set_postfix(loss=total_loss / max(1, n))
    if distributed:
        values = torch.tensor([meter.tp, meter.fp, meter.fn, total_loss, float(n)], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        meter.tp = float(values[0].item())
        meter.fp = float(values[1].item())
        meter.fn = float(values[2].item())
        total_loss = float(values[3].item())
        n = int(values[4].item())
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(1, n)
    return metrics


def _build_scheduler(optimizer, config):
    scheduler_cfg = config.get("training", {}).get("lr_scheduler")
    if not scheduler_cfg:
        return None
    if scheduler_cfg.get("type") == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.1)),
            patience=int(scheduler_cfg.get("patience", 3)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unknown lr_scheduler type: {scheduler_cfg.get('type')}")


def _amp_config(config, device: torch.device) -> tuple[bool, torch.dtype]:
    raw = config.get("training", {}).get("amp", {})
    enabled = bool(raw.get("enabled", False)) and device.type == "cuda"
    dtype_name = str(raw.get("dtype", "float16")).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return enabled, dtype


def _autocast_context(enabled: bool, dtype: torch.dtype, device: torch.device):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def _safe_segmentation_loss(
    logits: torch.Tensor | dict,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    config: dict | None = None,
) -> tuple[torch.Tensor, bool]:
    if isinstance(logits, dict):
        loss = segmentation_output_loss(logits, target, valid_mask, config or {})
        tensors = [value for value in logits.values() if torch.is_tensor(value)]
        finite_logits = all(torch.isfinite(value).all().item() for value in tensors)
    else:
        loss = segmentation_loss(logits.float(), target.float(), valid_mask.float())
        finite_logits = bool(torch.isfinite(logits).all().item())
    ok = bool(finite_logits and torch.isfinite(loss).item())
    return loss, ok


def _model_state_has_nonfinite(state_dict: dict[str, torch.Tensor]) -> bool:
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all().item():
            return True
    return False


def _model_has_nonfinite(model: torch.nn.Module) -> bool:
    return _model_state_has_nonfinite(model.state_dict())


def _clone_batchnorm_buffers(model: torch.nn.Module) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    snapshot = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            snapshot[name] = (
                module.running_mean.detach().clone() if module.running_mean is not None else None,
                module.running_var.detach().clone() if module.running_var is not None else None,
                module.num_batches_tracked.detach().clone() if hasattr(module, "num_batches_tracked") and module.num_batches_tracked is not None else None,
            )
    return snapshot


def _restore_batchnorm_buffers(
    model: torch.nn.Module,
    snapshot: dict[str, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]],
) -> None:
    modules = dict(model.named_modules())
    for name, (running_mean, running_var, num_batches_tracked) in snapshot.items():
        module = modules[name]
        if running_mean is not None:
            module.running_mean.copy_(running_mean)
        if running_var is not None:
            module.running_var.copy_(running_var)
        if num_batches_tracked is not None:
            module.num_batches_tracked.copy_(num_batches_tracked)


def _batchnorm_buffers_have_nonfinite(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for value in (module.running_mean, module.running_var):
                if value is not None and not torch.isfinite(value).all().item():
                    return True
    return False


def _batch_inputs_are_finite(batch: dict) -> bool:
    for key in ("image", "sar", "opt", "opt_mask", "mask", "valid_mask"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all().item():
            return False
    return True


def _sync_bad_flag(local_bad: bool, device: torch.device, dist_ctx: DistributedContext) -> bool:
    bad = torch.tensor([1 if local_bad else 0], device=device, dtype=torch.int)
    if dist_ctx.enabled:
        torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
    return bool(bad.item())


def _resume_checkpoint_path(config, output_dir: Path, explicit_resume: bool, explicit_path: str | None) -> Path | None:
    training = config.get("training", {})
    if explicit_path:
        return Path(explicit_path)
    resume_enabled = explicit_resume or bool(training.get("resume", False))
    if not resume_enabled:
        return None
    candidate = Path(training.get("resume_from") or output_dir / "checkpoints" / "last.ckpt")
    return candidate if candidate.exists() else None


def _build_dataset(config, root: Path, split: str, stats, training: bool):
    dataset_cfg = config["dataset"]
    cache_dir = dataset_cfg.get("cache_dir")
    exclude_scenes = dataset_cfg.get("exclude_scenes", [])
    if cache_dir and is_patch_cache_ready(cache_dir):
        return S1S2WaterPatchCacheDataset(cache_dir, split, exclude_scenes=exclude_scenes)
    patch_size = int(dataset_cfg.get("patch_size", 256))
    if training:
        return S1S2WaterDataset(
            root,
            split,
            patch_size=patch_size,
            stats=stats,
            training=True,
            patches_per_scene=int(dataset_cfg.get("patches_per_scene_per_epoch", 16)),
            stride=dataset_cfg.get("train_stride"),
            train_sampling=dataset_cfg.get("train_sampling", "random_crop"),
            seed=int(config["training"].get("seed", 4)),
            exclude_scenes=exclude_scenes,
        )
    return S1S2WaterDataset(
        root,
        split,
        patch_size=patch_size,
        stats=stats,
        training=False,
        stride=dataset_cfg.get("eval_stride"),
        exclude_scenes=exclude_scenes,
    )


def train_one_model(
    config,
    model_name: str,
    epochs_override: int | None = None,
    max_batches: int | None = None,
    output_dir_override: str | None = None,
    progress: bool = True,
    dist_ctx: DistributedContext | None = None,
    resume: bool = False,
    resume_from: str | None = None,
):
    dist_ctx = dist_ctx or DistributedContext(enabled=False)
    set_seed(int(config["training"].get("seed", 4)), bool(config["training"].get("deterministic", True)))
    output_dir = ensure_dir(output_dir_override or config.get("output_dir", "outputs/s1s2_water/6band_main"))
    config["output_dir"] = str(output_dir)
    if dist_ctx.is_main:
        for sub in ["checkpoints", "logs", "metrics", "figures/qualitative_clean", "figures/qualitative_degraded", "figures/metric_curves", "reports"]:
            ensure_dir(output_dir / sub)
    if dist_ctx.enabled:
        torch.distributed.barrier()
    logger = setup_file_logger(f"train.{model_name}", output_dir / "logs" / f"{model_name}_train.log") if dist_ctx.is_main else _null_logger(f"train.{model_name}.rank{dist_ctx.rank}")
    logger.info("Starting training")
    logger.info("model=%s output_dir=%s distributed=%s rank=%s world_size=%s", model_name, output_dir, dist_ctx.enabled, dist_ctx.rank, dist_ctx.world_size)
    root = _resolve_training_root(config)
    if dist_ctx.is_main:
        stats = load_or_create_stats(config, root, output_dir)
    if dist_ctx.enabled:
        torch.distributed.barrier()
    stats = json.loads(_stats_path(output_dir).read_text(encoding="utf-8"))
    patch_size = int(config["dataset"].get("patch_size", 256))
    train_ds = _build_dataset(config, root, "train", stats, training=True)
    val_ds = _build_dataset(config, root, "val", stats, training=False)
    batch_size = int(config["training"].get("batch_size", 16))
    train_sampler = DistributedSampler(train_ds, num_replicas=dist_ctx.world_size, rank=dist_ctx.rank, shuffle=True, drop_last=False) if dist_ctx.enabled else None
    val_sampler = DistributedSampler(val_ds, num_replicas=dist_ctx.world_size, rank=dist_ctx.rank, shuffle=False, drop_last=False) if dist_ctx.enabled else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=int(config["training"].get("num_workers", 0)), collate_fn=segmentation_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, num_workers=int(config["training"].get("num_workers", 0)), collate_fn=segmentation_collate)
    device = torch.device(f"cuda:{dist_ctx.local_rank}" if torch.cuda.is_available() and dist_ctx.enabled else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled, amp_dtype = _amp_config(config, device)
    logger.info(
        "device=%s amp_enabled=%s amp_dtype=%s train_samples=%s val_samples=%s train_batches=%s val_batches=%s batch_size_per_rank=%s effective_batch_size=%s train_sampling=%s patch_size=%s",
        device,
        amp_enabled,
        amp_dtype,
        len(train_ds),
        len(val_ds),
        len(train_loader),
        len(val_loader),
        batch_size,
        batch_size * dist_ctx.world_size,
        config["dataset"].get("train_sampling", "random_crop"),
        patch_size,
    )
    model = build_model(model_name, config).to(device)
    if dist_ctx.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_ctx.local_rank] if torch.cuda.is_available() else None,
            output_device=dist_ctx.local_rank if torch.cuda.is_available() else None,
        )
    target_model = model.module if dist_ctx.enabled else model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 0.001)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    scheduler = _build_scheduler(optimizer, config)
    epochs = epochs_override or int(config["training"].get("epochs", 100))
    early_cfg = build_early_stopping_config(config)
    early_state = EarlyStoppingState()
    start_epoch = 1
    final_metrics = {}
    resume_path = _resume_checkpoint_path(config, output_dir, resume, resume_from)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        if _model_state_has_nonfinite(checkpoint["model"]):
            raise RuntimeError(
                f"Refusing to resume from checkpoint with non-finite model state: {resume_path}. "
                "Use a finite checkpoint such as best.ckpt, or start a fresh output directory."
            )
        final_metrics = dict(checkpoint.get("metrics", {}))
        target_model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if amp_enabled and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("early_state"):
            early_state = EarlyStoppingState(**checkpoint["early_state"])
        else:
            metric = checkpoint.get("metrics", {})
            if "val_IoU" in metric:
                early_state = EarlyStoppingState(best_score=float(metric["val_IoU"]), best_epoch=int(checkpoint.get("epoch", 0)), stale=0)
        metric = checkpoint.get("metrics", {})
        resume_monitor_values = {}
        if "val_IoU" in metric:
            resume_monitor_values["val_IoU"] = float(metric["val_IoU"])
        if "val_F1" in metric:
            resume_monitor_values["val_F1"] = float(metric["val_F1"])
        if "val_loss" in metric:
            resume_monitor_values["val_loss"] = -float(metric["val_loss"])
        resume_score = resume_monitor_values.get(early_cfg.monitor)
        if resume_score is not None and torch.isfinite(torch.tensor(resume_score)) and resume_score > early_state.best_score:
            early_state = EarlyStoppingState(best_score=resume_score, best_epoch=int(checkpoint.get("epoch", 0)), stale=0)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        logger.info("Resumed training from %s at start_epoch=%d", resume_path, start_epoch)
    logger.info(
        "early_stopping enabled=%s monitor=%s patience=%s min_epochs=%s min_delta=%s",
        early_cfg.enabled,
        early_cfg.monitor,
        early_cfg.patience,
        early_cfg.min_epochs,
        early_cfg.min_delta,
    )
    log_rows = []
    epoch_metrics_file = output_dir / "metrics" / f"{model_name}_epoch_metrics.csv"
    summary_file = output_dir / "logs" / f"{model_name}_training_summary.json"
    epoch_iter = tqdm(range(start_epoch, epochs + 1), total=max(0, epochs - start_epoch + 1), desc=f"{model_name} epochs", dynamic_ncols=True, disable=(not progress) or (not dist_ctx.is_main))
    start_time = time.time()
    for epoch in epoch_iter:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n = 0
        skipped_batches = 0
        train_total = min(len(train_loader), max_batches) if max_batches is not None else len(train_loader)
        train_iter = tqdm(train_loader, total=train_total, desc=f"epoch {epoch} train", leave=False, dynamic_ncols=True, disable=(not progress) or (not dist_ctx.is_main))
        for batch_idx, batch in enumerate(train_iter):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            if config.get("robust_fusion", {}).get("enabled", False):
                batch = _apply_controlled_optical_missing(batch, config)
            elif model_name in {
                "mask_guided_late_fusion_unet",
                "mask_guided_late_fusion_sar_aux_unet",
                "mask_aware_cross_attention_fusion_unet",
                "mask_aware_cross_attention_no_sar_aux_unet",
                "mask_aware_cross_attention_fusion_unet_deep",
                "smagnet",
            }:
                batch["opt_mask"] = torch.ones_like(batch["opt"][:, :1])
            if _sync_bad_flag(not _batch_inputs_are_finite(batch), device, dist_ctx):
                skipped_batches += 1
                if dist_ctx.is_main and skipped_batches <= 5:
                    logger.warning("Skipping non-finite input batch epoch=%d batch=%d", epoch, batch_idx)
                continue
            bn_snapshot = _clone_batchnorm_buffers(target_model)
            with _autocast_context(amp_enabled, amp_dtype, device):
                output = model(select_model_input(batch, model_name))
                logits = resolve_segmentation_logits(output, batch, model_inference_mode(config))
            loss, loss_ok = _safe_segmentation_loss(output, batch["mask"], batch["valid_mask"], config) if not torch.is_tensor(output) else _safe_segmentation_loss(logits, batch["mask"], batch["valid_mask"], config)
            local_bad = (not loss_ok) or _batchnorm_buffers_have_nonfinite(target_model)
            if _sync_bad_flag(local_bad, device, dist_ctx):
                _restore_batchnorm_buffers(target_model, bn_snapshot)
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                if dist_ctx.is_main and skipped_batches <= 5:
                    logger.warning("Skipping non-finite forward/loss batch epoch=%d batch=%d loss_ok=%s", epoch, batch_idx, loss_ok)
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            n += 1
            train_iter.set_postfix(loss=train_loss / max(1, n), lr=optimizer.param_groups[0]["lr"])
        train_values = torch.tensor([train_loss, float(n), float(skipped_batches)], dtype=torch.float64, device=device)
        if dist_ctx.enabled:
            torch.distributed.all_reduce(train_values, op=torch.distributed.ReduceOp.SUM)
        train_loss_avg = float(train_values[0].item()) / max(1, int(train_values[1].item()))
        train_skipped = int(train_values[2].item())
        val_metrics = evaluate_epoch(model, val_loader, model_name, device, config, max_batches=max_batches, progress=progress and dist_ctx.is_main, epoch=epoch, distributed=dist_ctx.enabled, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss_avg,
            "val_loss": val_metrics["loss"],
            "val_IoU": val_metrics["IoU"],
            "val_F1": val_metrics["F1"],
            "val_Precision": val_metrics["Precision"],
            "val_Recall": val_metrics["Recall"],
            "epoch_seconds": time.time() - epoch_start,
        }
        final_metrics = row
        log_rows.append(row)
        if dist_ctx.is_main:
            append_epoch_metrics(epoch_metrics_file, row)
        monitor_values = {"val_IoU": val_metrics["IoU"], "val_F1": val_metrics["F1"], "val_loss": -val_metrics["loss"]}
        if early_cfg.monitor not in monitor_values:
            raise ValueError(f"Unknown early stopping monitor: {early_cfg.monitor}")
        early_state, improved, should_stop = update_early_stopping(early_state, monitor_values[early_cfg.monitor], epoch, early_cfg)
        has_bad_state = _sync_bad_flag(_model_has_nonfinite(target_model), device, dist_ctx)
        state_dict = model.module.state_dict() if dist_ctx.enabled else model.state_dict()
        checkpoint = {
            "model": state_dict,
            "config": config,
            "stats": stats,
            "model_name": model_name,
            "epoch": epoch,
            "metrics": row,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if amp_enabled else None,
            "early_state": {"best_score": early_state.best_score, "best_epoch": early_state.best_epoch, "stale": early_state.stale},
        }
        if dist_ctx.is_main and not has_bad_state:
            torch.save(checkpoint, output_dir / "checkpoints" / "last.ckpt")
            torch.save(checkpoint, output_dir / "checkpoints" / f"{model_name}_last.pt")
        elif dist_ctx.is_main:
            logger.error("Non-finite model state detected at epoch=%d; not overwriting last checkpoint.", epoch)
        if improved and dist_ctx.is_main and not has_bad_state:
            torch.save(checkpoint, output_dir / "checkpoints" / f"{model_name}_best.pt")
            torch.save(checkpoint, output_dir / "checkpoints" / "best.ckpt")
        if scheduler is not None:
            scheduler.step(val_metrics["IoU"])
        if dist_ctx.is_main:
            epoch_iter.set_postfix(best_score=early_state.best_score, val_IoU=val_metrics["IoU"], stale=early_state.stale, lr=optimizer.param_groups[0]["lr"])
            logger.info(
                "epoch=%d/%d lr=%.6g train_loss=%.6f val_loss=%.6f val_IoU=%.6f val_F1=%.6f val_Precision=%.6f val_Recall=%.6f best_score=%.6f best_epoch=%d stale=%d skipped_batches=%d early_stop_enabled=%s epoch_seconds=%.1f",
                epoch,
                epochs,
                current_lr,
                train_loss_avg,
                val_metrics["loss"],
                val_metrics["IoU"],
                val_metrics["F1"],
                val_metrics["Precision"],
                val_metrics["Recall"],
                early_state.best_score,
                early_state.best_epoch,
                early_state.stale,
                train_skipped,
                early_cfg.enabled,
                row["epoch_seconds"],
            )
            write_training_summary(
                summary_file,
                {
                    "model": model_name,
                    "output_dir": str(output_dir),
                    "device": str(device),
                    "distributed": {"enabled": dist_ctx.enabled, "world_size": dist_ctx.world_size},
                    "epochs_requested": epochs,
                    "epochs_completed": epoch,
                    "start_epoch": start_epoch,
                    "resume_from": str(resume_path) if resume_path is not None else None,
                    "early_stopping": {
                        "enabled": early_cfg.enabled,
                        "monitor": early_cfg.monitor,
                        "patience": early_cfg.patience,
                        "min_epochs": early_cfg.min_epochs,
                        "min_delta": early_cfg.min_delta,
                        "stale": early_state.stale,
                    },
                    "best_epoch": early_state.best_epoch,
                    "best_score": early_state.best_score,
                    "best_IoU": early_state.best_score if early_cfg.monitor == "val_IoU" else None,
                    "last_epoch_metrics": row,
                    "elapsed_seconds": time.time() - start_time,
                    "best_checkpoint": str(output_dir / "checkpoints" / "best.ckpt"),
                    "last_checkpoint": str(output_dir / "checkpoints" / "last.ckpt"),
                    "epoch_metrics_csv": str(epoch_metrics_file),
                    "text_log": str(output_dir / "logs" / f"{model_name}_train.log"),
                },
            )
        if dist_ctx.enabled:
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int)
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            logger.info("Early stopping triggered at epoch=%d after stale=%d", epoch, early_state.stale)
            break
    if dist_ctx.is_main:
        log_file = output_dir / "logs" / f"{model_name}_train_log.csv"
        if log_rows:
            with log_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
        else:
            logger.info("No epochs were run; checkpoint epoch is already >= requested epochs.")
        logger.info("Training finished best_score=%.6f best_epoch=%d elapsed_seconds=%.1f", early_state.best_score, early_state.best_epoch, time.time() - start_time)
    return output_dir / "checkpoints" / "best.ckpt", final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    dist_ctx = init_distributed_from_env()
    try:
        config = load_config(args.config)
        model_name = resolve_model_name(config, args.model)
        ckpt, metrics = train_one_model(config, model_name, args.epochs, args.max_batches, args.output_dir, progress=not args.no_progress, dist_ctx=dist_ctx, resume=args.resume, resume_from=args.resume_from)
        if dist_ctx.is_main:
            print(json.dumps({"checkpoint": str(ckpt), "metrics": metrics}, indent=2))
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
