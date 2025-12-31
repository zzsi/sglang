# SPDX-License-Identifier: Apache-2.0
"""
Pipeline configuration for HunyuanVideo-Avatar.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
    HunyuanConfig,
    clip_postprocess_text,
    clip_preprocess_text,
    llama_postprocess_text,
    llama_preprocess_text,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo_avatar import (
    HunyuanVideoAvatarConfig,
)


@dataclass
class HunyuanAvatarConfig(HunyuanConfig):
    """Configuration for HunyuanVideo-Avatar pipeline.

    Extends HunyuanConfig with:
    - Avatar-specific DiT config (with audio adapter parameters)
    - Audio encoder configuration (Whisper)
    """

    # Override task type - Avatar is essentially audio+image to video
    task_type: ModelTaskType = ModelTaskType.TI2V

    # Override DiT config to use avatar version
    dit_config: DiTConfig = field(default_factory=HunyuanVideoAvatarConfig)

    # Avatar-specific: flow_shift tuned for avatar generation
    flow_shift: int = 7

    # Audio encoder configuration
    # Whisper encoder for audio-to-embedding conversion
    # Can be HuggingFace model ID (e.g., "openai/whisper-tiny") or local path
    audio_encoder_path: str = "openai/whisper-tiny"

    # Whether to load audio encoder (can be disabled if using pre-computed embeddings)
    load_audio_encoder: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Ensure VAE encoder is loaded for reference image encoding
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Prepare avatar-specific conditioning kwargs for the transformer.

        Returns ref_latents and audio_embeds from batch.extra if available.
        """
        kwargs = {}

        # Get reference latents from batch.extra
        if "ref_latents" in batch.extra:
            ref_latents = batch.extra["ref_latents"]
            if ref_latents is not None:
                kwargs["ref_latents"] = ref_latents.to(device=device, dtype=dtype)

        # Get audio embeddings from batch.extra
        if "audio_embeds" in batch.extra:
            audio_embeds = batch.extra["audio_embeds"]
            if audio_embeds is not None:
                kwargs["audio_embeds"] = audio_embeds.to(device=device, dtype=dtype)

        return kwargs

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Prepare avatar-specific conditioning kwargs for negative CFG pass.

        For avatar generation, we still pass ref_latents but may want to
        handle audio differently for CFG.
        """
        kwargs = {}

        # Reference latents are still needed for negative pass
        if "ref_latents" in batch.extra:
            ref_latents = batch.extra["ref_latents"]
            if ref_latents is not None:
                kwargs["ref_latents"] = ref_latents.to(device=device, dtype=dtype)

        # For negative pass, we could either:
        # 1. Still use audio (current approach - keeps lip sync)
        # 2. Use zeros (would make CFG affect audio conditioning)
        # Using audio for now to maintain lip sync
        if "audio_embeds" in batch.extra:
            audio_embeds = batch.extra["audio_embeds"]
            if audio_embeds is not None:
                kwargs["audio_embeds"] = audio_embeds.to(device=device, dtype=dtype)

        return kwargs
