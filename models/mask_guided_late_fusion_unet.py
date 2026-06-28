from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import Encoder, UpBlock
from .output_utils import resolve_segmentation_logits


class MaskGuidedFusionBlock(nn.Module):
    """Lightweight channel-wise mask-aware gated SAR-optical fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.opt_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_sar: torch.Tensor, f_opt: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        mask_l = F.interpolate(opt_mask, size=f_sar.shape[-2:], mode="nearest")
        f_opt_masked = f_opt * mask_l
        opt_proj = self.opt_proj(f_opt_masked)
        gate = self.gate(torch.cat([f_sar, opt_proj, mask_l], dim=1))
        gate = gate * mask_l
        return f_sar + gate * opt_proj


class MaskGuidedLateFusionUNet(nn.Module):
    """Mask-guided late fusion U-Net for missing optical modality simulation.

    This follows the project's custom lightweight dual-encoder U-Net used by
    `LateFusionUNet`; it is not an EfficientNet-B0 encoder replica.
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.sar_encoder = Encoder(2, base_channels)
        self.opt_encoder = Encoder(4, base_channels)
        ch = self.sar_encoder.out_channels
        self.fusion_blocks = nn.ModuleList([MaskGuidedFusionBlock(c) for c in ch])
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)

    def forward(
        self,
        sar: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        opt: torch.Tensor | None = None,
        opt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(sar, (tuple, list)):
            sar, opt, opt_mask = sar
        if opt is None:
            raise ValueError("MaskGuidedLateFusionUNet requires optical input.")
        if opt_mask is None:
            opt_mask = torch.ones_like(opt[:, :1])

        sar_feats = self.sar_encoder(sar)
        opt_feats = self.opt_encoder(opt)
        fused = [
            block(f_sar, f_opt, opt_mask)
            for block, f_sar, f_opt in zip(self.fusion_blocks, sar_feats, opt_feats)
        ]
        x = self.up3(fused[3], fused[2])
        x = self.up2(x, fused[1])
        x = self.up1(x, fused[0])
        return self.head(x)


class MaskGuidedLateFusionSarAuxUNet(nn.Module):
    """Ablation: mask-guided late fusion with SAR auxiliary fallback, no cross-attention."""

    def __init__(self, base_channels: int = 32, inference_mode: str = "adaptive_fallback") -> None:
        super().__init__()
        self.inference_mode = inference_mode
        self.sar_encoder = Encoder(2, base_channels)
        self.opt_encoder = Encoder(4, base_channels)
        ch = self.sar_encoder.out_channels
        self.fusion_blocks = nn.ModuleList([MaskGuidedFusionBlock(c) for c in ch])
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)
        self.sar_up3 = UpBlock(ch[3], ch[2], ch[2])
        self.sar_up2 = UpBlock(ch[2], ch[1], ch[1])
        self.sar_up1 = UpBlock(ch[1], ch[0], ch[0])
        self.sar_head = nn.Conv2d(ch[0], 1, kernel_size=1)

    def forward(
        self,
        sar: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        opt: torch.Tensor | None = None,
        opt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(sar, (tuple, list)):
            sar, opt, opt_mask = sar
        if opt is None:
            raise ValueError("MaskGuidedLateFusionSarAuxUNet requires optical input.")
        if opt_mask is None:
            opt_mask = torch.ones_like(opt[:, :1])

        sar_feats = self.sar_encoder(sar)
        opt_feats = self.opt_encoder(opt)
        fused = [
            block(f_sar, f_opt, opt_mask)
            for block, f_sar, f_opt in zip(self.fusion_blocks, sar_feats, opt_feats)
        ]
        x = self.up3(fused[3], fused[2])
        x = self.up2(x, fused[1])
        x = self.up1(x, fused[0])
        logits_fused = self.head(x)

        sar_x = self.sar_up3(sar_feats[3], sar_feats[2])
        sar_x = self.sar_up2(sar_x, sar_feats[1])
        sar_x = self.sar_up1(sar_x, sar_feats[0])
        logits_sar = self.sar_head(sar_x)

        output = {"logits_fused": logits_fused, "logits_sar": logits_sar, "opt_mask": opt_mask}
        output["logits"] = resolve_segmentation_logits(output, {"opt_mask": opt_mask}, self.inference_mode)
        return output
