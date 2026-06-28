from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
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


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        ch = self.encoder.out_channels
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        x = self.up3(feats[3], feats[2])
        x = self.up2(x, feats[1])
        x = self.up1(x, feats[0])
        return self.head(x)


class LateFusionUNet(nn.Module):
    def __init__(self, base_channels: int = 32, gated: bool = False) -> None:
        super().__init__()
        self.gated = gated
        self.sar_encoder = Encoder(2, base_channels)
        self.opt_encoder = Encoder(4, base_channels)
        ch = self.sar_encoder.out_channels
        fused = [c * 2 for c in ch]
        if gated:
            self.gates = nn.ModuleList([nn.Conv2d(c * 2, c, kernel_size=1) for c in ch])
            fused = ch
        self.reduce3 = nn.Conv2d(fused[3], ch[3], kernel_size=1)
        self.reduce2 = nn.Conv2d(fused[2], ch[2], kernel_size=1)
        self.reduce1 = nn.Conv2d(fused[1], ch[1], kernel_size=1)
        self.reduce0 = nn.Conv2d(fused[0], ch[0], kernel_size=1)
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])
        self.head = nn.Conv2d(ch[0], 1, kernel_size=1)

    def _fuse(self, sar_feats: list[torch.Tensor], opt_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        fused = []
        for idx, (sar, opt) in enumerate(zip(sar_feats, opt_feats)):
            both = torch.cat([sar, opt], dim=1)
            if self.gated:
                gate = torch.sigmoid(self.gates[idx](both))
                fused.append(sar + gate * opt)
            else:
                fused.append(both)
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sar = x[:, :2]
        opt = x[:, 2:]
        fused = self._fuse(self.sar_encoder(sar), self.opt_encoder(opt))
        f0 = self.reduce0(fused[0])
        f1 = self.reduce1(fused[1])
        f2 = self.reduce2(fused[2])
        f3 = self.reduce3(fused[3])
        x = self.up3(f3, f2)
        x = self.up2(x, f1)
        x = self.up1(x, f0)
        return self.head(x)
