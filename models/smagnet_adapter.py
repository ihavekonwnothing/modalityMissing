from __future__ import annotations

import torch
import torch.nn as nn

from .SMAGnet import SMAGNet
from .output_utils import resolve_segmentation_logits


class SMAGNetAdapter(nn.Module):
    """Adapter that makes the external SMAGNet implementation fit this project.

    The original SMAGNet returns fused logits, SAR logits, and gate maps. This
    adapter exposes the same dict-style output contract used by the project's
    SAR-fallback models.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights_sar: str | None = None,
        encoder_weights_msi: str | None = None,
        decoder_channels: list[int] | None = None,
        decoder_use_batchnorm: bool = False,
        decoder_attention_type: str | None = None,
        sarmsiff_method: str = "sar_msi_gated",
        enable_spatial_mask: bool = True,
        inference_mode: str = "adaptive_fallback",
    ) -> None:
        super().__init__()
        self.inference_mode = inference_mode
        self.model = SMAGNet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights_sar=encoder_weights_sar,
            encoder_weights_msi=encoder_weights_msi,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels or [256, 128, 64, 32, 16],
            decoder_attention_type=decoder_attention_type,
            classes=1,
            activation=None,
            sarmsiff_method=sarmsiff_method,
            enable_spatial_mask=enable_spatial_mask,
        )

    def forward(
        self,
        sar: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        opt: torch.Tensor | None = None,
        opt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor | None]]:
        if isinstance(sar, (tuple, list)):
            sar, opt, opt_mask = sar
        if opt is None:
            raise ValueError("SMAGNetAdapter requires optical input.")
        if opt_mask is None:
            opt_mask = torch.ones_like(opt[:, :1])

        # SMAGNet expects an invalid-region mask: valid=0, invalid=1.
        spatial_mask = 1.0 - opt_mask
        logits_fused, logits_sar, gate_map = self.model(sar, opt, spatial_mask=spatial_mask)
        output: dict[str, torch.Tensor | list[torch.Tensor | None]] = {
            "logits_fused": logits_fused,
            "logits_sar": logits_sar,
            "opt_mask": opt_mask,
            "gate_map": gate_map,
        }
        output["logits"] = resolve_segmentation_logits(output, {"opt_mask": opt_mask}, self.inference_mode)
        return output
