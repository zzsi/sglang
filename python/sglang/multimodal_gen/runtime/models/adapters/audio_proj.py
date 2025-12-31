# SPDX-License-Identifier: Apache-2.0
"""
Audio Projection Model for HunyuanVideo-Avatar.

This module projects audio embeddings (from Whisper or similar) into context tokens
that can be injected into the DiT transformer for audio-conditioned video generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class AudioProjNet2(nn.Module):
    """
    Audio Projection Network that transforms audio embeddings into context tokens.

    Takes audio embeddings from a speech encoder (e.g., Whisper) and projects them
    into a sequence of context tokens that can be used for cross-attention in the
    video generation transformer.

    Args:
        seq_len: Length of audio sequence window (default: 5)
        blocks: Number of encoder blocks in audio features (default: 12 for Whisper)
        channels: Channel dimension of audio features (default: 768)
        intermediate_dim: Hidden dimension of projection MLP (default: 512)
        output_dim: Output dimension matching DiT hidden size (default: 768)
        context_tokens: Number of context tokens per frame (default: 4)
        dtype: Data type for parameters

    Input shape:
        audio_embeds: (batch, num_frames, seq_len, blocks, channels)

    Output shape:
        context_tokens: (batch, num_frames, context_tokens, output_dim)
    """

    def __init__(
        self,
        seq_len: int = 5,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 4,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        factory_kwargs = {"dtype": dtype}

        # 3-layer MLP projection
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim, **factory_kwargs)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim, **factory_kwargs)
        self.proj3 = nn.Linear(
            intermediate_dim, context_tokens * output_dim, **factory_kwargs
        )

        self.norm = nn.LayerNorm(output_dim, **factory_kwargs)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        """
        Project audio embeddings to context tokens.

        Args:
            audio_embeds: Audio embeddings with shape
                (batch, num_frames, seq_len, blocks, channels)

        Returns:
            Context tokens with shape (batch, num_frames, context_tokens, output_dim)
        """
        # Get video length before reshaping
        video_length = audio_embeds.shape[1]

        # Flatten batch and frames: (batch * frames, seq_len, blocks, channels)
        audio_embeds = rearrange(audio_embeds, "b f w l c -> (b f) w l c")
        batch_size = audio_embeds.shape[0]

        # Flatten to (batch * frames, seq_len * blocks * channels)
        audio_embeds = audio_embeds.view(batch_size, -1)

        # MLP projection with ReLU activations
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        # Project to context tokens
        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        # Layer normalization
        context_tokens = self.norm(context_tokens)

        # Reshape back to (batch, frames, context_tokens, output_dim)
        context_tokens = rearrange(
            context_tokens, "(b f) m c -> b f m c", f=video_length
        )

        return context_tokens
