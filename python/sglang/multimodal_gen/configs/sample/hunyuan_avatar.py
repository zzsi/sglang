# SPDX-License-Identifier: Apache-2.0
"""
Sampling parameters for HunyuanVideo-Avatar.
"""
from dataclasses import dataclass, field

import numpy as np
import PIL.Image
import torch

from sglang.multimodal_gen.configs.sample.hunyuan import HunyuanSamplingParams


@dataclass
class HunyuanAvatarSamplingParams(HunyuanSamplingParams):
    """Sampling parameters for HunyuanVideo-Avatar audio-driven generation."""

    # Override defaults for avatar generation
    num_inference_steps: int = 50
    num_frames: int = 129  # Must be 4k+1 for temporal compression
    height: int = 704
    width: int = 768
    fps: int = 24

    # Avatar-specific parameters
    # Reference image for the avatar (required)
    ref_image_path: str | None = None

    # Audio input (one of these is required for avatar generation)
    audio_path: str | None = None
    # Pre-computed audio embeddings from Whisper
    # Shape: (num_frames, seq_len, blocks, channels)
    audio_embeds: np.ndarray | None = None

    # Motion control (optional, not implemented in Phase 2)
    # motion_exp: float | None = None  # Expression intensity
    # motion_pose: float | None = None  # Pose intensity

    # Avatar-specific supported resolutions
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            # Portrait-oriented resolutions (recommended for avatars)
            (768, 704),   # ~1:1
            (704, 768),   # ~1:1
            (768, 1024),  # 3:4
            (1024, 768),  # 4:3
            # 720p resolutions
            (1280, 720),  # 16:9
            (720, 1280),  # 9:16
        ]
    )

    def _validate(self):
        """Validate avatar-specific parameters."""
        super()._validate()

        # Reference image is required
        if self.ref_image_path is None:
            raise ValueError(
                "ref_image_path is required for HunyuanVideo-Avatar generation"
            )

        # Audio input is required (either path or pre-computed embeddings)
        if self.audio_path is None and self.audio_embeds is None:
            raise ValueError(
                "Either audio_path or audio_embeds must be provided for "
                "HunyuanVideo-Avatar generation"
            )

    def get_extra_fields(self) -> dict:
        """Get avatar-specific fields to be stored in batch.extra.

        These fields are not part of the standard Req dataclass and need
        to be passed separately to the pipeline stages.
        """
        extra = {}

        # Load reference image
        if self.ref_image_path is not None:
            ref_image = PIL.Image.open(self.ref_image_path).convert("RGB")
            extra["ref_image"] = ref_image
            extra["ref_image_path"] = self.ref_image_path

        # Audio path or embeddings
        if self.audio_path is not None:
            extra["audio_path"] = self.audio_path

        if self.audio_embeds is not None:
            # Convert numpy array to tensor if needed
            if isinstance(self.audio_embeds, np.ndarray):
                extra["audio_embeds"] = torch.from_numpy(self.audio_embeds)
            else:
                extra["audio_embeds"] = self.audio_embeds

        return extra
