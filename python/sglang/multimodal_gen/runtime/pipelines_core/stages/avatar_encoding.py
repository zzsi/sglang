# SPDX-License-Identifier: Apache-2.0
"""
Avatar-specific encoding stages for HunyuanVideo-Avatar pipeline.

This module contains stages for encoding reference images and audio
for avatar generation.
"""

import PIL
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vision_utils import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ReferenceImageEncodingStage(PipelineStage):
    """
    Stage for encoding reference images into latent space for avatar generation.

    This stage encodes the reference (portrait) image using VAE and stores
    the result in batch.extra["ref_latents"].
    """

    def __init__(self, vae: ParallelTiledVAE, **kwargs) -> None:
        super().__init__()
        self.vae: ParallelTiledVAE = vae

    def load_model(self):
        self.vae = self.vae.to(get_local_torch_device())

    def offload_model(self):
        if self.server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

    def preprocess(
        self,
        image: torch.Tensor | PIL.Image.Image,
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        """Preprocess reference image to target size."""
        if isinstance(image, PIL.Image.Image):
            # Resize to target dimensions
            image = image.resize(
                (target_width, target_height), PIL.Image.Resampling.LANCZOS
            )
            image = pil_to_numpy(image)
            image = numpy_to_pt(image)

        # Normalize if needed
        do_normalize = True
        if image.min() < 0:
            do_normalize = False
        if do_normalize:
            image = normalize(image)

        return image

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode reference image into latent space.

        Args:
            batch: The current batch information.
                   Expects batch.extra["ref_image"] to contain PIL Image or tensor.
            server_args: The inference arguments.

        Returns:
            The batch with batch.extra["ref_latents"] populated.
        """
        # Get reference image from batch.extra or condition_image
        ref_image = batch.extra.get("ref_image", batch.condition_image)
        if ref_image is None:
            logger.warning("No reference image provided for avatar generation")
            return batch

        self.load_model()

        # Get target dimensions
        target_height = batch.height
        target_width = batch.width

        # Preprocess reference image
        image = self.preprocess(ref_image, target_height, target_width)
        image = image.to(get_local_torch_device(), dtype=torch.float32)

        # Add batch and temporal dimensions: (C, H, W) -> (B, C, 1, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch
        image = image.unsqueeze(2)  # Add temporal

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Encode reference image
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                image = image.to(vae_dtype)

            latent_dist: DiagonalGaussianDistribution = self.vae.encode(image)
            if isinstance(latent_dist, AutoencoderKLOutput):
                latent_dist = latent_dist.latent_dist

        # Sample from latent distribution
        generator = batch.generator
        if generator is None:
            raise ValueError("Generator must be provided")

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if sample_mode == "sample":
            ref_latents = latent_dist.sample(generator)
        elif sample_mode == "argmax":
            ref_latents = latent_dist.mode()
        else:
            raise ValueError(f"Unknown sample mode: {sample_mode}")

        # Apply scale and shift
        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=ref_latents.device,
                dtype=ref_latents.dtype,
                vae=self.vae,
            )
        )

        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(ref_latents.device)
        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(ref_latents.device)

        ref_latents = (ref_latents - shift_factor) * scaling_factor

        # Store in batch.extra
        # ref_latents shape: (B, C, 1, H, W)
        batch.extra["ref_latents"] = ref_latents

        logger.debug(f"Reference image encoded to latents with shape: {ref_latents.shape}")

        self.offload_model()
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify reference image encoding stage inputs."""
        result = VerificationResult()
        # Check that either ref_image or condition_image is available
        ref_image = batch.extra.get("ref_image", batch.condition_image)
        result.add_check(
            "ref_image",
            ref_image,
            lambda x: x is not None,
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify reference image encoding stage outputs."""
        result = VerificationResult()
        result.add_check(
            "ref_latents",
            batch.extra.get("ref_latents"),
            lambda x: x is not None and isinstance(x, torch.Tensor),
        )
        return result


class WhisperAudioEncoder:
    """
    Whisper-based audio encoder for HunyuanVideo-Avatar.

    Encodes audio files into embeddings compatible with the avatar transformer.
    Output shape: (batch, num_frames, seq_len=10, blocks=5, channels=384)
    """

    def __init__(self, whisper_path: str):
        """
        Initialize Whisper encoder.

        Args:
            whisper_path: Path to Whisper model (e.g., "openai/whisper-tiny" or local path)
        """
        from transformers import WhisperModel, AutoFeatureExtractor

        self.whisper = WhisperModel.from_pretrained(whisper_path)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)
        self.device = None

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.whisper = self.whisper.to(device)
        return self

    def encode(
        self,
        audio_path: str,
        num_frames: int,
        fps: float = 25.0,
    ) -> torch.Tensor:
        """
        Encode audio file to embeddings.

        Args:
            audio_path: Path to audio file
            num_frames: Number of video frames to generate
            fps: Video frame rate (25 or 12.5)

        Returns:
            Audio embeddings with shape (1, num_frames, 10, 5, 384)
        """
        import librosa

        # Load audio at 16kHz
        audio_input, sr = librosa.load(audio_path, sr=16000)
        assert sr == 16000, f"Expected 16kHz, got {sr}Hz"

        # Extract mel features using Whisper's feature extractor
        # Process in chunks of 750*640 samples (30 seconds)
        window = 750 * 640
        audio_features = []
        for i in range(0, len(audio_input), window):
            chunk = audio_input[i:i + window]
            features = self.feature_extractor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
            ).input_features
            audio_features.append(features)

        audio_features = torch.cat(audio_features, dim=-1)

        # Encode with Whisper encoder (take first 3000 frames = 30s max)
        audio_features = audio_features[:, :, :3000].to(
            self.device, dtype=self.whisper.dtype
        )

        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(
                audio_features,
                output_hidden_states=True,
            )
            hidden_states = encoder_outputs.hidden_states

        # Stack hidden states: (batch, seq, layers, hidden)
        audio_feats = torch.stack(hidden_states, dim=2)

        # Pad with zeros at the beginning (matching original implementation)
        audio_feats = torch.cat(
            [torch.zeros_like(audio_feats[:, :4]), audio_feats], dim=1
        )

        # Extract per-frame audio embeddings
        # For 25 fps: step_ts = 1, for 12.5 fps: step_ts = 2
        step_ts = 1 if fps == 25 else 2

        audio_prompts = []
        for f in range(num_frames):
            cur_t = f * step_ts * 2  # Multiply by 2 for audio-video alignment
            # Window of 10 timesteps
            audio_clip = audio_feats[:, cur_t:cur_t + 10]
            if audio_clip.shape[1] < 10:
                # Pad if needed
                pad = torch.zeros(
                    audio_clip.shape[0],
                    10 - audio_clip.shape[1],
                    *audio_clip.shape[2:],
                    device=audio_clip.device,
                    dtype=audio_clip.dtype,
                )
                audio_clip = torch.cat([audio_clip, pad], dim=1)
            audio_prompts.append(audio_clip)

        # Stack: (batch, num_frames, 10, layers, hidden)
        audio_prompts = torch.stack(audio_prompts, dim=1)

        # Select/pad to 5 layers (Whisper-tiny has 4 layers)
        num_layers = audio_prompts.shape[3]
        if num_layers < 5:
            pad_layers = 5 - num_layers
            last_layer = audio_prompts[:, :, :, -1:, :]
            audio_prompts = torch.cat(
                [audio_prompts, last_layer.repeat(1, 1, 1, pad_layers, 1)], dim=3
            )
        else:
            audio_prompts = audio_prompts[:, :, :, :5, :]

        # Final shape: (batch, num_frames, 10, 5, 384)
        return audio_prompts


class AudioEncodingStage(PipelineStage):
    """
    Stage for encoding audio into embeddings for avatar generation.

    This stage either:
    1. Uses pre-computed audio embeddings from batch.extra["audio_embeds"]
    2. Encodes audio from batch.extra["audio_path"] using Whisper

    The audio embeddings are stored in batch.extra["audio_embeds"].
    """

    def __init__(self, audio_encoder=None, whisper_path: str | None = None, **kwargs) -> None:
        """
        Initialize audio encoding stage.

        Args:
            audio_encoder: Optional pre-initialized audio encoder
            whisper_path: Path to Whisper model (used if audio_encoder is None)
        """
        super().__init__()
        self.audio_encoder = audio_encoder
        self.whisper_path = whisper_path
        self._whisper_encoder = None

    def _get_whisper_encoder(self) -> WhisperAudioEncoder | None:
        """Lazily initialize Whisper encoder."""
        if self._whisper_encoder is not None:
            return self._whisper_encoder

        if self.audio_encoder is not None:
            return self.audio_encoder

        if self.whisper_path is not None:
            logger.info(f"Loading Whisper encoder from {self.whisper_path}")
            self._whisper_encoder = WhisperAudioEncoder(self.whisper_path)
            return self._whisper_encoder

        return None

    def load_model(self):
        encoder = self._get_whisper_encoder()
        if encoder is not None:
            encoder.to(get_local_torch_device())

    def offload_model(self):
        if self._whisper_encoder is not None and self.server_args.vae_cpu_offload:
            self._whisper_encoder.to("cpu")

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode audio or use pre-computed embeddings.

        Args:
            batch: The current batch information.
                   Can have batch.extra["audio_embeds"] (pre-computed) or
                   batch.extra["audio_path"] (to be encoded).
            server_args: The inference arguments.

        Returns:
            The batch with batch.extra["audio_embeds"] populated.
        """
        # Check if pre-computed embeddings are provided
        if "audio_embeds" in batch.extra and batch.extra["audio_embeds"] is not None:
            audio_embeds = batch.extra["audio_embeds"]
            if isinstance(audio_embeds, torch.Tensor):
                batch.extra["audio_embeds"] = audio_embeds.to(get_local_torch_device())
                logger.debug(
                    f"Using pre-computed audio embeddings with shape: {audio_embeds.shape}"
                )
                return batch

        # Try to encode from audio path
        audio_path = batch.extra.get("audio_path")
        if audio_path is None:
            logger.warning("No audio input provided for avatar generation")
            return batch

        # Get or create Whisper encoder
        encoder = self._get_whisper_encoder()
        if encoder is None:
            raise ValueError(
                "Audio path provided but no audio encoder available. "
                "Either provide pre-computed audio_embeds, pass whisper_path, "
                "or configure audio encoder."
            )

        self.load_model()

        # Get encoding parameters
        num_frames = batch.num_frames
        fps = batch.extra.get("fps", 25.0)

        try:
            with torch.no_grad():
                audio_embeds = encoder.encode(audio_path, num_frames, fps)

            batch.extra["audio_embeds"] = audio_embeds
            logger.info(f"Audio encoded to embeddings with shape: {audio_embeds.shape}")

        except ImportError as e:
            raise ImportError(
                f"Missing dependency for audio encoding: {e}. "
                "Install with: pip install librosa transformers"
            )
        except Exception as e:
            logger.error(f"Failed to encode audio: {e}")
            raise

        self.offload_model()
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify audio encoding stage inputs."""
        result = VerificationResult()
        # Either audio_embeds or audio_path should be provided
        has_audio = (
            batch.extra.get("audio_embeds") is not None
            or batch.extra.get("audio_path") is not None
        )
        result.add_check("audio_input", has_audio, lambda x: x)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify audio encoding stage outputs."""
        result = VerificationResult()
        result.add_check(
            "audio_embeds",
            batch.extra.get("audio_embeds"),
            lambda x: x is not None and isinstance(x, torch.Tensor),
        )
        return result
