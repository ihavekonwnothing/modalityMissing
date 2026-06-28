from __future__ import annotations

from typing import Any, Dict

from .mask_aware_cross_attention_fusion_unet import MaskAwareCrossAttentionFusionUNet, MaskAwareCrossAttentionNoSarAuxUNet
from .mask_aware_cross_attention_fusion_unet_deep import MaskAwareCrossAttentionFusionUNetDeep
from .mask_guided_late_fusion_unet import MaskGuidedLateFusionUNet, MaskGuidedLateFusionSarAuxUNet
from .smagnet_adapter import SMAGNetAdapter
from .unet_baselines import S1OnlyUNet, S2OnlyUNet, UNetEfficientNetB0
from .unet import LateFusionUNet, UNet


def build_model(name: str, config: Dict[str, Any] | None = None):
    config = config or {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    base_channels = int(config.get("model_base_channels", 32))
    if name == "s1_only_unet":
        if model_cfg.get("encoder") == "efficientnet-b0":
            return S1OnlyUNet(num_classes=int(model_cfg.get("num_classes", 1)), encoder_weights=model_cfg.get("encoder_weights"))
        return UNet(in_channels=2, base_channels=base_channels)
    if name == "s2_only_unet":
        if model_cfg.get("encoder") == "efficientnet-b0":
            return S2OnlyUNet(num_classes=int(model_cfg.get("num_classes", 1)), encoder_weights=model_cfg.get("encoder_weights"))
        return UNet(in_channels=4, base_channels=base_channels)
    if name == "unet_efficientnet_b0":
        return UNetEfficientNetB0(
            in_channels=int(model_cfg.get("in_channels", config.get("in_channels", 3))),
            num_classes=int(model_cfg.get("num_classes", 1)),
            encoder_weights=model_cfg.get("encoder_weights"),
        )
    if name == "early_fusion_unet":
        return UNet(in_channels=6, base_channels=base_channels)
    if name == "late_fusion_unet":
        return LateFusionUNet(base_channels=base_channels, gated=False)
    if name == "proposed_robust_fusion_unet":
        return LateFusionUNet(base_channels=base_channels, gated=True)
    if name == "mask_guided_late_fusion_unet":
        return MaskGuidedLateFusionUNet(base_channels=base_channels)
    if name == "mask_guided_late_fusion_sar_aux_unet":
        return MaskGuidedLateFusionSarAuxUNet(
            base_channels=base_channels,
            inference_mode=str(model_cfg.get("inference_mode", "adaptive_fallback")),
        )
    if name == "mask_aware_cross_attention_fusion_unet":
        return MaskAwareCrossAttentionFusionUNet(
            base_channels=base_channels,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            window_size=int(model_cfg.get("window_size", 8)),
            inference_mode=str(model_cfg.get("inference_mode", "adaptive_fallback")),
        )
    if name == "mask_aware_cross_attention_no_sar_aux_unet":
        return MaskAwareCrossAttentionNoSarAuxUNet(
            base_channels=base_channels,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            window_size=int(model_cfg.get("window_size", 8)),
        )
    if name == "mask_aware_cross_attention_fusion_unet_deep":
        return MaskAwareCrossAttentionFusionUNetDeep(
            base_channels=base_channels,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            window_size=int(model_cfg.get("window_size", 8)),
            inference_mode=str(model_cfg.get("inference_mode", "adaptive_fallback")),
        )
    if name == "smagnet":
        return SMAGNetAdapter(
            encoder_name=str(model_cfg.get("encoder_name", model_cfg.get("encoder", "resnet34"))),
            encoder_depth=int(model_cfg.get("encoder_depth", 5)),
            encoder_weights_sar=model_cfg.get("encoder_weights_sar"),
            encoder_weights_msi=model_cfg.get("encoder_weights_msi"),
            decoder_channels=model_cfg.get("decoder_channels"),
            decoder_use_batchnorm=bool(model_cfg.get("decoder_use_batchnorm", False)),
            decoder_attention_type=model_cfg.get("decoder_attention_type"),
            sarmsiff_method=str(model_cfg.get("sarmsiff_method", "sar_msi_gated")),
            enable_spatial_mask=bool(model_cfg.get("enable_spatial_mask", True)),
            inference_mode=str(model_cfg.get("inference_mode", "adaptive_fallback")),
        )
    raise ValueError(f"Unknown model: {name}")


def select_model_input(batch, model_name: str):
    if model_name == "s1_only_unet":
        return batch["sar"]
    if model_name == "s2_only_unet":
        return batch["opt"]
    if model_name in {
        "mask_guided_late_fusion_unet",
        "mask_guided_late_fusion_sar_aux_unet",
        "mask_aware_cross_attention_fusion_unet",
        "mask_aware_cross_attention_no_sar_aux_unet",
        "mask_aware_cross_attention_fusion_unet_deep",
        "smagnet",
    }:
        return batch["sar"], batch["opt"], batch.get("opt_mask")
    return batch["image"]
