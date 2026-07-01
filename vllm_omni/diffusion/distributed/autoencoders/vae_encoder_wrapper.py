# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight VAE encoder wrapper for disaggregated inference.

This wrapper loads only the encoder part of a VAE model, saving memory
when the decoder is not needed (e.g., in a dedicated encode_image stage).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class VAEEncoderWrapper(nn.Module):
    """Wrapper that exposes only the encoder half of a VAE.

    Loads the full VAE checkpoint but only keeps encoder-related parameters
    in memory.  The decoder weights are discarded after extraction.
    """

    def __init__(
        self,
        vae: nn.Module,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

        # Extract encoder components from the full VAE.
        # Different VAE architectures use different attribute names.
        if hasattr(vae, "encoder"):
            self.encoder = vae.encoder
        else:
            raise ValueError("VAE does not have an 'encoder' attribute.")

        if hasattr(vae, "quant_conv"):
            self.quant_conv = vae.quant_conv
        else:
            # Some VAEs don't have quant_conv (e.g., latent diffusion models)
            self.quant_conv = nn.Identity()

        # Store config for downstream consumers
        self.config = getattr(vae, "config", None)

        # Move to device
        self.to(self.device)

        # Free decoder weights to save memory
        self._free_decoder_weights(vae)

    def _free_decoder_weights(self, vae: nn.Module) -> None:
        """Delete decoder-related parameters from the original VAE."""
        decoder_attrs = ["decoder", "post_quant_conv"]
        for attr in decoder_attrs:
            if hasattr(vae, attr):
                delattr(vae, attr)
        logger.info("VAEEncoderWrapper: freed decoder weights")

    def encode(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> Any:
        """Encode input tensor to latent space.

        Args:
            x: Input tensor (e.g., image or video frames).

        Returns:
            Latent representation.  The exact type depends on the VAE
            architecture (may be a tuple, dict, or tensor).
        """
        h = self.encoder(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        h = self.quant_conv(h)
        return h
