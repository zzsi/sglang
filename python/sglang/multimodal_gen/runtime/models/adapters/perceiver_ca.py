# SPDX-License-Identifier: Apache-2.0
"""
Perceiver Cross-Attention for HunyuanVideo-Avatar.

This module implements cross-attention between audio context tokens and
visual latents, allowing audio features to modulate the video generation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PerceiverAttentionCA(nn.Module):
    """
    Perceiver-style Cross-Attention for audio-to-visual conditioning.

    Implements cross-attention where:
    - Query (Q) comes from visual latents
    - Key (K) and Value (V) come from audio context tokens

    This allows audio information to be injected into specific layers
    of the video generation transformer.

    Args:
        dim: Hidden dimension of the model (default: 3072)
        dim_head: Dimension per attention head (default: 1024)
        heads: Number of attention heads (default: 1, single-head attention)
        dtype: Data type for parameters

    Note:
        The output projection is zero-initialized for residual learning,
        allowing the model to start with identity mapping.
    """

    def __init__(
        self,
        dim: int = 3072,
        dim_head: int = 1024,
        heads: int = 1,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        # Inner dimension (single head in original implementation)
        inner_dim = dim_head

        factory_kwargs = {"dtype": dtype}

        # Layer norms for both inputs
        # Named norm1/norm2 to match checkpoint naming
        self.norm1 = nn.LayerNorm(dim, **factory_kwargs)  # For audio
        self.norm2 = nn.LayerNorm(dim, **factory_kwargs)  # For latents

        # Projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False, **factory_kwargs)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False, **factory_kwargs)
        self.to_out = nn.Linear(inner_dim, dim, bias=False, **factory_kwargs)

        # Zero-initialize output projection for residual learning
        nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        audio_context: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply cross-attention from latents to audio context.

        Args:
            audio_context: Audio context tokens with shape
                (batch, num_frames, num_tokens, dim) or (batch, num_tokens, dim)
            latents: Visual latent features with shape
                (batch, num_frames, seq_len, dim) or (batch, seq_len, dim)

        Returns:
            Output features with same shape as latents, to be added as residual
        """
        # Normalize inputs
        audio_context = self.norm1(audio_context)
        latents = self.norm2(latents)

        # Compute Q from latents, K/V from audio
        q = self.to_q(latents)
        kv = self.to_kv(audio_context)
        k, v = kv.chunk(2, dim=-1)

        # Scaled dot-product attention
        # Using double sqrt for numerical stability with fp16
        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        attn_weights = (q * scale) @ (k * scale).transpose(-2, -1)
        attn_weights = torch.softmax(attn_weights.float(), dim=-1).type(q.dtype)

        # Apply attention to values
        out = attn_weights @ v

        # Project output
        return self.to_out(out)
