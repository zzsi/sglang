# SPDX-License-Identifier: Apache-2.0
"""
Avatar-specific encoding stages for HunyuanVideo-Avatar pipeline.

This module contains stages for encoding reference images and audio
for avatar generation.
"""

from typing import Optional

import numpy as np
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


class LLaVATextEncoder:
    """
    LLaVA-based text encoder for HunyuanVideo-Avatar.

    Encodes text + reference image into hidden states using LLaVA-LLaMA-3-8B.
    The reference image provides facial identity information to the text embeddings.
    """

    # Prompt template matching original HunyuanVideo-Avatar
    PROMPT_TEMPLATE = (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    )
    CROP_START = 95  # Remove instruction tokens, keep only prompt tokens

    def __init__(self, model_path: str, precision: str = "fp16"):
        """
        Initialize LLaVA encoder.

        Args:
            model_path: Path to LLaVA model
            precision: Model precision ("fp16", "bf16", or "fp32")
        """
        from transformers import LlavaForConditionalGeneration, LlamaTokenizerFast

        self.model_path = model_path
        self.precision = precision
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

        logger.info(f"Loading LLaVA encoder from {model_path}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        # Get the final layer norm for hidden state processing
        # The structure varies: language_model.model.norm or language_model.norm
        if hasattr(self.model.language_model, 'model'):
            self.final_layer_norm = self.model.language_model.model.norm
        else:
            self.final_layer_norm = self.model.language_model.norm

        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            model_path,
            padding_side="right",
        )

        # Image processor for LLaVA's CLIP vision encoder
        from transformers import AutoProcessor
        try:
            processor = AutoProcessor.from_pretrained(model_path)
            self.image_processor = processor.image_processor
        except Exception:
            from transformers import CLIPImageProcessor
            self.image_processor = CLIPImageProcessor.from_pretrained(model_path)

        self.device = None

        # Number of image tokens that LLaVA uses (576 patches for 336x336 with patch_size=14)
        # Plus some additional tokens for separators = 575
        self.num_image_tokens = 575

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self

    def preprocess_image(self, image: PIL.Image.Image) -> torch.Tensor:
        """
        Preprocess image for LLaVA's CLIP vision encoder.

        Matches original HunyuanVideo-Avatar preprocessing:
        - Resize to 336x336 (bilinear)
        - ToTensor (0-255 -> 0-1)
        - Normalize with CLIP mean/std

        Args:
            image: PIL Image (reference portrait)

        Returns:
            Tensor of shape (1, 3, 336, 336)
        """
        from torchvision import transforms

        # Exact transform from original HunyuanVideo-Avatar
        llava_transform = transforms.Compose([
            transforms.Resize(
                (336, 336),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.4082107),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        # Apply transform and add batch dimension
        pixel_values = llava_transform(image).unsqueeze(0)

        return pixel_values.to(self.device, dtype=self.dtype)

    def encode(
        self,
        text: str,
        image: Optional[PIL.Image.Image] = None,
        max_length: int = 256,
        hidden_state_skip_layer: int = 2,
        apply_final_norm: bool = True,
    ) -> torch.Tensor:
        """
        Encode text (and optionally image) to hidden states.

        Args:
            text: Text prompt
            image: Optional reference image for identity injection
            max_length: Maximum token length
            hidden_state_skip_layer: Number of layers to skip from the end (0 = last layer)
            apply_final_norm: Whether to apply final layer norm to intermediate layers

        Returns:
            Hidden states tensor of shape (batch, seq_len, hidden_dim=4096)
        """
        # Apply prompt template
        formatted_text = self.PROMPT_TEMPLATE.format(text)

        # Add image placeholder if image is provided
        # Following original HunyuanVideo-Avatar pattern
        if image is not None:
            formatted_text = formatted_text + "\nThe person looks like<image>"

        # Tokenize
        text_inputs = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            if image is not None:
                # Process image and get vision features manually
                pixel_values = self.preprocess_image(image)

                # Get image features from vision encoder
                image_outputs = self.model.vision_tower(
                    pixel_values,
                    output_hidden_states=True
                )
                # Use the second-to-last layer as per vision_feature_layer=-2
                selected_image_feature = image_outputs.hidden_states[-2]

                # Project through multi-modal projector
                image_features = self.model.multi_modal_projector(selected_image_feature)
                # image_features shape: (1, 576, 4096)

                # Get text embeddings
                inputs_embeds = self.model.get_input_embeddings()(input_ids)

                # Find <image> token position and replace with image features
                image_token_id = self.model.config.image_token_index  # 128257
                image_token_mask = input_ids == image_token_id

                # Build the combined embeddings
                # Replace the single <image> token with 576 image patch embeddings
                batch_size = input_ids.shape[0]
                new_embeds_list = []
                new_attention_list = []

                for b in range(batch_size):
                    image_positions = torch.where(image_token_mask[b])[0]
                    if len(image_positions) > 0:
                        pos = image_positions[0].item()
                        # Before image token
                        before = inputs_embeds[b, :pos]
                        before_mask = attention_mask[b, :pos]
                        # Image features
                        img_feats = image_features[b]  # (576, 4096)
                        img_mask = torch.ones(img_feats.shape[0], device=self.device, dtype=attention_mask.dtype)
                        # After image token (skip the <image> token itself)
                        after = inputs_embeds[b, pos + 1:]
                        after_mask = attention_mask[b, pos + 1:]
                        # Concatenate
                        new_embed = torch.cat([before, img_feats, after], dim=0)
                        new_mask = torch.cat([before_mask, img_mask, after_mask], dim=0)
                    else:
                        new_embed = inputs_embeds[b]
                        new_mask = attention_mask[b]

                    new_embeds_list.append(new_embed)
                    new_attention_list.append(new_mask)

                # Stack and potentially truncate/pad to max length
                max_len = max(e.shape[0] for e in new_embeds_list)
                final_embeds = torch.zeros(batch_size, max_len, inputs_embeds.shape[-1],
                                          device=self.device, dtype=inputs_embeds.dtype)
                final_attention = torch.zeros(batch_size, max_len,
                                             device=self.device, dtype=attention_mask.dtype)

                for b, (emb, mask) in enumerate(zip(new_embeds_list, new_attention_list)):
                    seq_len = min(emb.shape[0], max_len)
                    final_embeds[b, :seq_len] = emb[:seq_len]
                    final_attention[b, :seq_len] = mask[:seq_len]

                # Forward through language model with inputs_embeds
                outputs = self.model.language_model(
                    inputs_embeds=final_embeds,
                    attention_mask=final_attention,
                    output_hidden_states=True,
                )
            else:
                # Text-only encoding through language model
                outputs = self.model.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states
        if hidden_state_skip_layer > 0:
            last_hidden_state = hidden_states[-(hidden_state_skip_layer + 1)]
            # Apply final layer norm for intermediate layers
            if apply_final_norm:
                last_hidden_state = self.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = hidden_states[-1]

        # Crop instruction tokens (keep only prompt tokens)
        if self.CROP_START > 0:
            last_hidden_state = last_hidden_state[:, self.CROP_START:]

        return last_hidden_state


class CLIPTextEncoder:
    """
    CLIP text encoder for pooled embeddings.

    Used alongside LLaVA to provide pooled text representations.
    """

    def __init__(self, model_path: str, precision: str = "fp16"):
        """
        Initialize CLIP text encoder.

        Args:
            model_path: Path to CLIP model
            precision: Model precision
        """
        from transformers import CLIPTextModel, CLIPTokenizer

        self.model_path = model_path
        self.precision = precision
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

        logger.info(f"Loading CLIP text encoder from {model_path}")
        self.model = CLIPTextModel.from_pretrained(model_path)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = self.model.to(self.dtype)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, max_length=77)
        self.device = None

    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to pooled embedding.

        Args:
            text: Text prompt

        Returns:
            Pooled embedding tensor of shape (batch, 768)
        """
        # Tokenize
        text_inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Return pooler output (CLS token embedding)
        return outputs.pooler_output


class AvatarTextEncodingStage(PipelineStage):
    """
    Combined text encoding stage for HunyuanVideo-Avatar.

    Uses LLaVA for main hidden states (with reference image injection)
    and CLIP for pooled embeddings.
    """

    def __init__(
        self,
        llava_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        precision: str = "fp16",
        **kwargs,
    ) -> None:
        """
        Initialize avatar text encoding stage.

        Args:
            llava_path: Path to LLaVA model
            clip_path: Path to CLIP model
            precision: Model precision
        """
        super().__init__()
        self.llava_path = llava_path
        self.clip_path = clip_path
        self.precision = precision
        self._llava_encoder: Optional[LLaVATextEncoder] = None
        self._clip_encoder: Optional[CLIPTextEncoder] = None

    def _get_llava_encoder(self) -> Optional[LLaVATextEncoder]:
        """Lazily initialize LLaVA encoder."""
        if self._llava_encoder is not None:
            return self._llava_encoder
        if self.llava_path is not None:
            self._llava_encoder = LLaVATextEncoder(self.llava_path, self.precision)
            return self._llava_encoder
        return None

    def _get_clip_encoder(self) -> Optional[CLIPTextEncoder]:
        """Lazily initialize CLIP encoder."""
        if self._clip_encoder is not None:
            return self._clip_encoder
        if self.clip_path is not None:
            self._clip_encoder = CLIPTextEncoder(self.clip_path, self.precision)
            return self._clip_encoder
        return None

    def load_model(self):
        llava = self._get_llava_encoder()
        if llava is not None:
            llava.to(get_local_torch_device())

        clip = self._get_clip_encoder()
        if clip is not None:
            clip.to(get_local_torch_device())

    def offload_model(self):
        if self._llava_encoder is not None:
            self._llava_encoder.to("cpu")
        if self._clip_encoder is not None:
            self._clip_encoder.to("cpu")
        torch.cuda.empty_cache()

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode text with LLaVA (+ reference image) and CLIP.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with prompt embeddings populated.
        """
        prompt = batch.prompt
        if prompt is None:
            prompt = ""

        # Get reference image for LLaVA
        ref_image = batch.extra.get("ref_image", batch.condition_image)

        self.load_model()

        # Encode with LLaVA
        llava = self._get_llava_encoder()
        if llava is not None:
            hidden_states = llava.encode(prompt, image=ref_image)
            batch.prompt_embeds.append(hidden_states)
            logger.debug(f"LLaVA hidden states shape: {hidden_states.shape}")

            # Encode negative prompt if CFG is enabled
            if batch.do_classifier_free_guidance:
                neg_prompt = batch.negative_prompt or ""
                # For negative, don't include image (or use zeros)
                neg_hidden_states = llava.encode(neg_prompt, image=None)
                if batch.negative_prompt_embeds is not None:
                    batch.negative_prompt_embeds.append(neg_hidden_states)

        # Encode with CLIP for pooled embeddings
        clip = self._get_clip_encoder()
        if clip is not None:
            pooled_embeds = clip.encode(prompt)
            batch.pooled_embeds.append(pooled_embeds)
            logger.debug(f"CLIP pooled embeds shape: {pooled_embeds.shape}")

            if batch.do_classifier_free_guidance:
                neg_prompt = batch.negative_prompt or ""
                neg_pooled_embeds = clip.encode(neg_prompt)
                batch.neg_pooled_embeds.append(neg_pooled_embeds)

        self.offload_model()
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify text encoding stage inputs."""
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, lambda x: x is None or isinstance(x, str))
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify text encoding stage outputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            lambda x: len(x) > 0,
        )
        return result


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
