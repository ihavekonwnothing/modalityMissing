from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_guided_late_fusion_unet import MaskGuidedFusionBlock
from .unet import ConvBlock, UpBlock
from .output_utils import resolve_segmentation_logits


class DeepEncoder(nn.Module):
    """Five-stage encoder that produces 1/1, 1/2, 1/4, 1/8, 1/16 features."""

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.blocks = nn.ModuleList()
        current = in_channels
        for ch in channels:
            self.blocks.append(ConvBlock(current, ch))
            current = ch
        self.pool = nn.MaxPool2d(2)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            feats.append(x)
            if idx != len(self.blocks) - 1:
                x = self.pool(x)
        return feats


class MaskAwareCrossAttentionBlock(nn.Module):
    """Windowed SAR-query / optical-key-value cross-attention fusion."""

    def __init__(
        self,
        sar_channels: int,
        opt_channels: int | None = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        opt_channels = int(opt_channels or sar_channels)
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.proj_q = nn.Conv2d(sar_channels, hidden_dim, kernel_size=1)
        self.proj_k = nn.Conv2d(opt_channels, hidden_dim, kernel_size=1)
        self.proj_v = nn.Conv2d(opt_channels, hidden_dim, kernel_size=1)
        self.proj_out = nn.Conv2d(hidden_dim, sar_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def _windows(self, x: torch.Tensor, window: int) -> tuple[torch.Tensor, tuple[int, int]]:
        b, c, h, w = x.shape
        pad_h = (window - h % window) % window
        pad_w = (window - w % window) % window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // window, window, wp // window, window)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, window * window, c), (hp, wp)

    def _unwindows(
        self,
        x: torch.Tensor,
        padded_hw: tuple[int, int],
        batch: int,
        channels: int,
        original_hw: tuple[int, int],
        window: int,
    ) -> torch.Tensor:
        hp, wp = padded_hw
        x = x.view(batch, hp // window, wp // window, window, window, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch, channels, hp, wp)
        return x[..., : original_hw[0], : original_hw[1]]

    def forward(self, f_sar: torch.Tensor, f_opt: torch.Tensor, opt_mask: torch.Tensor) -> torch.Tensor:
        b, _, h, w = f_sar.shape
        mask_l = F.interpolate(opt_mask, size=(h, w), mode="nearest")
        f_opt_masked = f_opt * mask_l
        q = self.proj_q(f_sar)
        k = self.proj_k(f_opt_masked)
        v = self.proj_v(f_opt_masked)
        window = max(1, min(self.window_size, h, w))
        q_w, padded_hw = self._windows(q, window)
        k_w, _ = self._windows(k, window)
        v_w, _ = self._windows(v, window)
        mask_w, _ = self._windows(mask_l, window)
        tokens = q_w.shape[1]
        head_dim = self.hidden_dim // self.num_heads
        q_w = q_w.view(-1, tokens, self.num_heads, head_dim).transpose(1, 2)
        k_w = k_w.view(-1, tokens, self.num_heads, head_dim).transpose(1, 2)
        v_w = v_w.view(-1, tokens, self.num_heads, head_dim).transpose(1, 2)
        attn = torch.matmul(q_w, k_w.transpose(-2, -1)) / math.sqrt(head_dim)
        valid_tokens = mask_w.squeeze(-1) > 0.5
        all_missing = ~valid_tokens.any(dim=-1, keepdim=True)
        valid_tokens = torch.where(all_missing, torch.ones_like(valid_tokens), valid_tokens)
        attn = attn.masked_fill(~valid_tokens[:, None, None, :], -1e4)
        attn = torch.softmax(attn, dim=-1)
        cross = torch.matmul(attn, v_w).transpose(1, 2).contiguous().view(-1, tokens, self.hidden_dim)
        cross = self._unwindows(cross, padded_hw, b, self.hidden_dim, (h, w), window)
        out = self.proj_out(cross)
        return f_sar + self.gamma.to(dtype=out.dtype) * out * mask_l


class MaskAwareCrossAttentionFusionUNetDeep(nn.Module):
    """Five-stage dual-encoder U-Net for 1/2, 1/4 gated fusion and 1/8, 1/16 cross-attention."""

    def __init__(
        self,
        base_channels: int = 32,
        hidden_dim: int = 128,
        num_heads: int = 4,
        window_size: int = 8,
        inference_mode: str = "adaptive_fallback",
    ) -> None:
        super().__init__()
        self.inference_mode = inference_mode
        self.sar_encoder = DeepEncoder(2, base_channels)
        self.opt_encoder = DeepEncoder(4, base_channels)
        ch = self.sar_encoder.out_channels

        self.stem_reduce = nn.Conv2d(ch[0] * 2, ch[0], kernel_size=1)
        self.fusion_blocks = nn.ModuleList(
            [
                MaskGuidedFusionBlock(ch[1]),
                MaskGuidedFusionBlock(ch[2]),
                MaskAwareCrossAttentionBlock(ch[3], ch[3], hidden_dim=hidden_dim, num_heads=num_heads, window_size=window_size),
                MaskAwareCrossAttentionBlock(ch[4], ch[4], hidden_dim=hidden_dim, num_heads=num_heads, window_size=window_size),
            ]
        )

        self.up4 = UpBlock(ch[4], ch[3], ch[3])
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)

        self.sar_up4 = UpBlock(ch[4], ch[3], ch[3])
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
            raise ValueError("MaskAwareCrossAttentionFusionUNetDeep requires optical input.")
        if opt_mask is None:
            opt_mask = torch.ones_like(opt[:, :1])

        sar_feats = self.sar_encoder(sar)
        opt_feats = self.opt_encoder(opt)

        fused0 = self.stem_reduce(torch.cat([sar_feats[0], opt_feats[0]], dim=1))
        fused1 = self.fusion_blocks[0](sar_feats[1], opt_feats[1], opt_mask)
        fused2 = self.fusion_blocks[1](sar_feats[2], opt_feats[2], opt_mask)
        fused3 = self.fusion_blocks[2](sar_feats[3], opt_feats[3], opt_mask)
        fused4 = self.fusion_blocks[3](sar_feats[4], opt_feats[4], opt_mask)
        fused = [fused0, fused1, fused2, fused3, fused4]

        x = self.up4(fused[4], fused[3])
        x = self.up3(x, fused[2])
        x = self.up2(x, fused[1])
        x = self.up1(x, fused[0])
        logits_fused = self.head(x)

        sar_x = self.sar_up4(sar_feats[4], sar_feats[3])
        sar_x = self.sar_up3(sar_x, sar_feats[2])
        sar_x = self.sar_up2(sar_x, sar_feats[1])
        sar_x = self.sar_up1(sar_x, sar_feats[0])
        logits_sar = self.sar_head(sar_x)

        output = {"logits_fused": logits_fused, "logits_sar": logits_sar, "opt_mask": opt_mask}
        output["logits"] = resolve_segmentation_logits(output, {"opt_mask": opt_mask}, self.inference_mode)
        return output
