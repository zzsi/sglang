# SPDX-License-Identifier: Apache-2.0
"""
HunyuanVideo-Avatar diffusion pipeline implementation.

This module contains the pipeline for generating talking head videos
from a reference image and audio input.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.avatar_encoding import (
    AudioEncodingStage,
    ReferenceImageEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class HunyuanVideoAvatarPipeline(ComposedPipelineBase):
    """
    Pipeline for HunyuanVideo-Avatar talking head generation.

    This pipeline generates video of a talking head from:
    - A reference portrait image
    - Audio input (or pre-computed audio embeddings)
    - Optional text prompt for style/context

    The pipeline stages are:
    1. Input validation
    2. Text encoding (for style/context prompts)
    3. Reference image encoding (VAE encoding of portrait)
    4. Audio encoding (Whisper embeddings or pre-computed)
    5. Conditioning preparation
    6. Timestep preparation
    7. Latent preparation
    8. Denoising (with audio-visual cross-attention)
    9. Decoding (VAE decode to video)
    """

    pipeline_name = "HunyuanVideoAvatarPipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # 1. Input validation
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        # 2. Text encoding (for style/context prompts)
        self.add_stage(
            stage_name="prompt_encoding_stage_primary",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )

        # 3. Reference image encoding (VAE encode portrait)
        self.add_stage(
            stage_name="reference_image_encoding_stage",
            stage=ReferenceImageEncodingStage(
                vae=self.get_module("vae"),
            ),
        )

        # 4. Audio encoding (Whisper or pre-computed)
        # Note: audio_encoder is optional - can use pre-computed embeddings
        audio_encoder = self.get_module("audio_encoder", required=False)
        self.add_stage(
            stage_name="audio_encoding_stage",
            stage=AudioEncodingStage(
                audio_encoder=audio_encoder,
            ),
        )

        # 5. Conditioning preparation
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        # 6. Timestep preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 7. Latent preparation
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # 8. Denoising (with audio cross-attention)
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 9. Decoding (VAE decode to video)
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(
                vae=self.get_module("vae"),
            ),
        )


# Entry class for registry
EntryClass = HunyuanVideoAvatarPipeline
