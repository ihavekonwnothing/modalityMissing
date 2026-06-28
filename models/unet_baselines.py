from __future__ import annotations

from typing import Optional
import warnings

import torch
import torch.nn as nn

from .unet import UNet


class UNetEfficientNetB0(nn.Module):
    """U-Net baseline with EfficientNet-B0 encoder when SMP is available.

    The S1S2-Water baseline uses U-Net with EfficientNet-B0 as the preferred
    encoder. If segmentation_models_pytorch is unavailable, this class falls
    back to the local lightweight U-Net and emits a clear warning because that
    fallback is not an exact EfficientNet-B0 reproduction.
    """

    def __init__(self, in_channels: int, num_classes: int = 1, encoder_weights: Optional[str] = None) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.encoder_name = "efficientnet-b0"
        self.uses_smp = False
        try:
            import segmentation_models_pytorch as smp

            self.model = smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=self.in_channels,
                classes=self.num_classes,
                activation=None,
            )
            self.uses_smp = True
        except ImportError:
            warnings.warn(
                "segmentation_models_pytorch is not installed; using a lightweight custom U-Net fallback. "
                "This is not an exact EfficientNet-B0 reproduction.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.model = UNet(in_channels=self.in_channels, base_channels=32)
            if self.num_classes != 1:
                self.model.head = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class S1OnlyUNet(UNetEfficientNetB0):
    """S1-only baseline, input order [VV, VH]."""

    def __init__(self, num_classes: int = 1, encoder_weights: Optional[str] = None) -> None:
        super().__init__(in_channels=2, num_classes=num_classes, encoder_weights=encoder_weights)


class S2OnlyUNet(UNetEfficientNetB0):
    """S2-only baseline, input order [Blue, Green, Red, NIR]."""

    def __init__(self, num_classes: int = 1, encoder_weights: Optional[str] = None) -> None:
        super().__init__(in_channels=4, num_classes=num_classes, encoder_weights=encoder_weights)
