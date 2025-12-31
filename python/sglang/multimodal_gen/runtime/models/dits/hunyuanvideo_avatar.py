# SPDX-License-Identifier: Apache-2.0
"""
HunyuanVideo-Avatar Transformer for audio-driven avatar generation.

This is a standalone implementation based on SGLang's HunyuanVideo transformer,
with additional components for audio conditioning and reference image injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.configs.models.dits.hunyuanvideo import (
    HunyuanVideoArchConfig,
    HunyuanVideoConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    get_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    PatchEmbed,
    TimestepEmbedder,
    unpatchify,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.models.adapters import AudioProjNet2, PerceiverAttentionCA
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin

# Import block classes from base HunyuanVideo
from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
    MMDoubleStreamBlock,
    MMSingleStreamBlock,
    SingleTokenRefiner,
    FinalLayer,
)


@dataclass
class HunyuanVideoAvatarArchConfig(HunyuanVideoArchConfig):
    """Architecture configuration for HunyuanVideo-Avatar model.

    Extends HunyuanVideoArchConfig with audio adapter parameters.
    """

    # Audio projection parameters (for AudioProjNet2)
    audio_seq_len: int = 10
    audio_blocks: int = 5
    audio_channels: int = 384
    audio_intermediate_dim: int = 1024
    audio_context_tokens: int = 4

    # Audio injection layers (indices of double_blocks where audio is injected)
    audio_injection_layers: list[int] = field(
        default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    )

    # PerceiverAttentionCA parameters
    perceiver_dim_head: int = 1024


@dataclass
class HunyuanVideoAvatarConfig(DiTConfig):
    """Configuration wrapper for HunyuanVideo-Avatar model."""

    arch_config: HunyuanVideoAvatarArchConfig = field(
        default_factory=HunyuanVideoAvatarArchConfig
    )
    prefix: str = "HunyuanAvatar"


class HunyuanVideoAvatarTransformer(CachableDiT, OffloadableDiTMixin):
    """
    HunyuanVideo-Avatar Transformer for audio-driven avatar generation.

    This model extends the base HunyuanVideo transformer with:
    - Reference image conditioning (ref_in)
    - Audio feature projection (audio_proj)
    - Audio injection via cross-attention at specific layers (audio_adapter_blocks)

    Args:
        config: HunyuanVideoAvatarConfig with model parameters
        hf_config: HuggingFace config dict for compatibility
    """

    # Use base HunyuanVideo class attributes (inherited from arch config)
    _fsdp_shard_conditions = HunyuanVideoAvatarArchConfig()._fsdp_shard_conditions
    _compile_conditions = HunyuanVideoAvatarArchConfig()._compile_conditions
    _supported_attention_backends = HunyuanVideoAvatarArchConfig()._supported_attention_backends
    param_names_mapping = HunyuanVideoAvatarArchConfig().param_names_mapping
    reverse_param_names_mapping = HunyuanVideoAvatarArchConfig().reverse_param_names_mapping
    lora_param_names_mapping = HunyuanVideoAvatarArchConfig().lora_param_names_mapping

    def __init__(self, config: HunyuanVideoAvatarConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)

        self.patch_size = [config.patch_size_t, config.patch_size, config.patch_size]
        self.in_channels = config.in_channels
        self.num_channels_latents = config.num_channels_latents
        self.out_channels = (
            config.in_channels if config.out_channels is None else config.out_channels
        )
        self.unpatchify_channels = self.out_channels
        self.guidance_embeds = config.guidance_embeds
        self.rope_dim_list = list(config.rope_axes_dim)
        self.rope_theta = config.rope_theta
        self.text_states_dim = config.text_embed_dim
        self.text_states_dim_2 = config.pooled_projection_dim
        self.dtype = config.dtype

        pe_dim = config.hidden_size // config.num_attention_heads
        if sum(config.rope_axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.rope_axes_dim} but expected positional dim {pe_dim}"
            )

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents

        # ==================== Base HunyuanVideo components ====================

        # Image projection
        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            dtype=config.dtype,
            prefix=f"{config.prefix}.img_in",
        )

        # Text projection
        self.txt_in = SingleTokenRefiner(
            self.text_states_dim,
            config.hidden_size,
            config.num_attention_heads,
            depth=config.num_refiner_layers,
            dtype=config.dtype,
            prefix=f"{config.prefix}.txt_in",
        )

        # Time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size,
            act_layer="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.time_in",
        )

        # Text modulation
        self.vector_in = MLP(
            self.text_states_dim_2,
            self.hidden_size,
            self.hidden_size,
            act_type="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.vector_in",
        )

        # Guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size,
                act_layer="silu",
                dtype=config.dtype,
                prefix=f"{config.prefix}.guidance_in",
            )
            if self.guidance_embeds
            else None
        )

        # Double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    dtype=config.dtype,
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=f"{config.prefix}.double_blocks.{i}",
                )
                for i in range(config.num_layers)
            ]
        )

        # Single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    dtype=config.dtype,
                    supported_attention_backends=self._supported_attention_backends,
                    prefix=f"{config.prefix}.single_blocks.{i + config.num_layers}",
                )
                for i in range(config.num_single_layers)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            config.hidden_size,
            self.patch_size,
            self.out_channels,
            dtype=config.dtype,
            prefix=f"{config.prefix}.final_layer",
        )

        # ==================== Avatar-specific components ====================

        # Reference image embedding (same architecture as img_in)
        self.ref_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            dtype=config.dtype,
            prefix=f"{config.prefix}.ref_in",
        )

        # Linear projection to combine reference + noise latents
        self.before_proj = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            params_dtype=config.dtype,
            prefix=f"{config.prefix}.before_proj",
        )

        # Audio projection
        self.audio_proj = AudioProjNet2(
            seq_len=config.audio_seq_len,
            blocks=config.audio_blocks,
            channels=config.audio_channels,
            intermediate_dim=config.audio_intermediate_dim,
            output_dim=self.hidden_size,
            context_tokens=config.audio_context_tokens,
            dtype=config.dtype,
        )

        # Audio injection configuration
        self.audio_injection_layers = config.audio_injection_layers
        self.audio_injection_set = set(self.audio_injection_layers)

        # Audio adapter blocks (one per injection layer)
        self.audio_adapter_blocks = nn.ModuleList(
            [
                PerceiverAttentionCA(
                    dim=self.hidden_size,
                    dim_head=config.perceiver_dim_head,
                    dtype=config.dtype,
                )
                for _ in range(len(self.audio_injection_layers))
            ]
        )

        # Mapping from layer index to adapter index
        self._layer_to_adapter = {
            layer_idx: adapter_idx
            for adapter_idx, layer_idx in enumerate(self.audio_injection_layers)
        }

        # ==================== Motion/FPS components ====================
        # These provide additional control over expression, pose, and frame rate

        # FPS conditioning (outputs full hidden_size = 3072)
        self.fps_proj = TimestepEmbedder(
            self.hidden_size,
            act_layer="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.fps_proj",
        )

        # Motion expression control (outputs hidden_size // 4 = 768)
        # Original checkpoint has: 256 -> 768 -> 768
        self.motion_exp = TimestepEmbedder(
            self.hidden_size // 4,  # 768
            act_layer="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.motion_exp",
        )

        # Motion pose control (outputs hidden_size // 4 = 768)
        self.motion_pose = TimestepEmbedder(
            self.hidden_size // 4,  # 768
            act_layer="silu",
            dtype=config.dtype,
            prefix=f"{config.prefix}.motion_pose",
        )


        self.__post_init__()
        self.layer_names = ["double_blocks", "single_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance: torch.Tensor | None = None,
        # Avatar-specific inputs
        ref_latents: torch.Tensor | None = None,
        audio_embeds: torch.Tensor | None = None,
        # Motion/FPS control inputs
        fps: torch.Tensor | None = None,
        motion_exp: torch.Tensor | None = None,
        motion_pose: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the HunyuanVideo-Avatar model.

        Args:
            hidden_states: Input video latents [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Diffusion timestep
            guidance: Guidance scale for CFG
            ref_latents: Reference image latents [B, C, T_ref, H, W] (typically T_ref >= 1)
            audio_embeds: Audio embeddings [B, num_frames, seq_len, blocks, channels]
            fps: Frame rate value [B] (default: 24.0)
            motion_exp: Expression intensity [B] (default: 1.0, range 0-2)
            motion_pose: Pose intensity [B] (default: 1.0, range 0-2)

        Returns:
            Denoised output tensor
        """
        if guidance is None:
            guidance = torch.tensor(
                [6016.0], device=hidden_states.device, dtype=hidden_states.dtype
            )

        img = x = hidden_states
        t = timestep

        # Split text embeddings
        if isinstance(encoder_hidden_states, torch.Tensor):
            txt = encoder_hidden_states[:, 1:]
            text_states_2 = encoder_hidden_states[:, 0, : self.text_states_dim_2]
        else:
            txt = encoder_hidden_states[0]
            text_states_2 = encoder_hidden_states[1]

        # Get spatial dimensions
        batch_size, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Get rotary embeddings - include extra frame for ref_latents_first
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            ((tt + 1) * get_sp_world_size(), th, tw),  # +1 for ref frame
            self.hidden_size,
            self.num_attention_heads,
            self.rope_dim_list,
            self.rope_theta,
        )
        freqs_cos = freqs_cos.to(x.device)
        freqs_sin = freqs_sin.to(x.device)

        # Prepare modulation vectors
        vec = self.time_in(t)
        vec = vec + self.vector_in(text_states_2)

        if self.guidance_in and guidance is not None:
            vec = vec + self.guidance_in(guidance)

        # Add motion/FPS modulation
        if fps is not None:
            vec = vec + self.fps_proj(fps)
        else:
            # Default FPS of 24
            default_fps = torch.tensor([24.0], device=hidden_states.device, dtype=hidden_states.dtype)
            default_fps = default_fps.expand(batch_size)
            vec = vec + self.fps_proj(default_fps)

        # Motion embeddings output 768-dim (hidden_size // 4)
        # We tile them 4x to 3072-dim to match vec
        if motion_exp is not None:
            motion_exp_emb = self.motion_exp(motion_exp)  # [B, 768]
        else:
            # Default expression intensity of 30.0 (matches original hardcoded value)
            default_exp = torch.tensor([30.0], device=hidden_states.device, dtype=hidden_states.dtype)
            default_exp = default_exp.expand(batch_size)
            motion_exp_emb = self.motion_exp(default_exp)  # [B, 768]
        # Tile to match hidden_size: [B, 768] -> [B, 3072]
        vec = vec + motion_exp_emb.repeat(1, 4)

        if motion_pose is not None:
            motion_pose_emb = self.motion_pose(motion_pose)  # [B, 768]
        else:
            # Default pose intensity of 25.0 (matches original hardcoded value)
            default_pose = torch.tensor([25.0], device=hidden_states.device, dtype=hidden_states.dtype)
            default_pose = default_pose.expand(batch_size)
            motion_pose_emb = self.motion_pose(default_pose)  # [B, 768]
        # Tile to match hidden_size: [B, 768] -> [B, 3072]
        vec = vec + motion_pose_emb.repeat(1, 4)

        # Embed image
        img = self.img_in(img)

        # Process reference image (following original HunyuanVideo-Avatar)
        ref_length = 0
        if ref_latents is not None:
            # Extract first frame of reference for concatenation
            ref_latents_first = ref_latents[:, :, :1].clone()

            # Process full ref_latents through ref_in
            ref_embedded = self.ref_in(ref_latents)

            # Process first ref frame through img_in (not ref_in!)
            ref_first_embedded = self.img_in(ref_latents_first)

            # Add projected reference to img
            ref_proj, _ = self.before_proj(ref_embedded)
            img = img + ref_proj

            # Concatenate ref_first to beginning of sequence
            # ref_first_embedded: [B, 1*th*tw, hidden]
            ref_length = ref_first_embedded.shape[1]
            img = torch.cat([ref_first_embedded, img], dim=1)

        # Embed text
        txt = self.txt_in(txt, t)
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        # Process audio embeddings if provided
        audio_context = None
        if audio_embeds is not None:
            # audio_embeds: [B, num_frames, seq_len, blocks, channels]
            # audio_context: [B, num_frames, context_tokens, hidden_size]
            audio_context = self.audio_proj(audio_embeds)

            # For perceiver cross-attention, audio frames must match video latent frames
            # If audio is at video frame rate (ot), downsample to latent rate (tt)
            audio_frames = audio_context.shape[1]
            if audio_frames > tt:
                # Downsample audio from video frame rate to latent frame rate
                # audio_context: [B, frames, tokens, dim]
                B_audio, num_audio_frames, num_tokens, audio_dim = audio_context.shape
                # Reshape for 1D interpolation: [B * tokens * dim, 1, frames]
                audio_context = audio_context.permute(0, 2, 3, 1)  # [B, tokens, dim, frames]
                audio_context = audio_context.reshape(B_audio * num_tokens * audio_dim, 1, num_audio_frames)
                audio_context = F.interpolate(
                    audio_context.float(), size=tt, mode='linear', align_corners=False
                ).to(audio_context.dtype)
                # Reshape back: [B*tokens*dim, 1, tt] -> [B, tt, tokens, dim]
                audio_context = audio_context.reshape(B_audio, num_tokens, audio_dim, tt)
                audio_context = audio_context.permute(0, 3, 1, 2)  # [B, tt, tokens, dim]

        # Process through double stream blocks
        for layer_idx, block in enumerate(self.double_blocks):
            img, txt = block(img, txt, vec, freqs_cis)

            # Inject audio at specified layers
            if audio_context is not None and layer_idx in self.audio_injection_set:
                adapter_idx = self._layer_to_adapter[layer_idx]
                adapter = self.audio_adapter_blocks[adapter_idx]

                # Separate reference frame from video frames
                # img shape: [B, ref_length + tt*th*tw, hidden]
                if ref_length > 0:
                    img_ref = img[:, :ref_length]  # Reference frame tokens
                    img_video = img[:, ref_length:]  # Video frame tokens
                else:
                    img_ref = None
                    img_video = img

                # Reshape video tokens for cross-attention: [B, tt*th*tw, hidden] -> [B, tt, th*tw, hidden]
                img_video_reshaped = img_video.view(batch_size, tt, th * tw, self.hidden_size)

                # Apply audio cross-attention only to video (not reference)
                audio_delta = adapter(audio_context, img_video_reshaped)
                audio_delta = audio_delta.view(batch_size, -1, self.hidden_size)

                # Add audio delta to video tokens only (zero delta for ref)
                if ref_length > 0:
                    ref_delta = torch.zeros_like(img_ref)
                    img = img + torch.cat([ref_delta, audio_delta], dim=1)
                else:
                    img = img + audio_delta

        # Merge txt and img for single stream blocks
        x = torch.cat((img, txt), 1)

        # Process through single stream blocks
        for block in self.single_blocks:
            x = block(x, vec, txt_seq_len, freqs_cis)

        # Extract image features (excluding text)
        img = x[:, :img_seq_len, ...]

        # Remove reference frame tokens before final layer
        if ref_length > 0:
            img = img[:, ref_length:]

        # Final layer processing
        img = self.final_layer(img, vec)

        # Unpatchify to get original shape
        img = unpatchify(img, tt, th, tw, self.patch_size, self.out_channels)

        return img


    @staticmethod
    def remap_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Remap checkpoint keys from original HunyuanVideo-Avatar format to SGLang format.

        The original checkpoint uses different naming conventions:
        - MLP layers: fc1/fc2 vs fc_in/fc_out
        - Sequential layers: indexed (0, 2) vs named (fc_in, fc_out)
        - txt_in structure: individual_token_refiner.blocks vs refiner_blocks

        Args:
            state_dict: Original checkpoint state dict

        Returns:
            Remapped state dict compatible with SGLang model
        """
        remapped = {}

        for key, value in state_dict.items():
            new_key = key

            # time_in uses indexed naming (0, 2) -> (fc_in, fc_out)
            new_key = new_key.replace('time_in.mlp.0.', 'time_in.mlp.fc_in.')
            new_key = new_key.replace('time_in.mlp.2.', 'time_in.mlp.fc_out.')

            # fps_proj uses indexed naming
            new_key = new_key.replace('fps_proj.mlp.0.', 'fps_proj.mlp.fc_in.')
            new_key = new_key.replace('fps_proj.mlp.2.', 'fps_proj.mlp.fc_out.')

            # motion_exp uses indexed naming
            new_key = new_key.replace('motion_exp.mlp.0.', 'motion_exp.mlp.fc_in.')
            new_key = new_key.replace('motion_exp.mlp.2.', 'motion_exp.mlp.fc_out.')

            # motion_pose uses indexed naming
            new_key = new_key.replace('motion_pose.mlp.0.', 'motion_pose.mlp.fc_in.')
            new_key = new_key.replace('motion_pose.mlp.2.', 'motion_pose.mlp.fc_out.')

            # txt_in.t_embedder uses indexed naming
            new_key = new_key.replace('txt_in.t_embedder.mlp.0.', 'txt_in.t_embedder.mlp.fc_in.')
            new_key = new_key.replace('txt_in.t_embedder.mlp.2.', 'txt_in.t_embedder.mlp.fc_out.')

            # txt_in structure
            new_key = new_key.replace('txt_in.c_embedder.linear_1.', 'txt_in.c_embedder.fc_in.')
            new_key = new_key.replace('txt_in.c_embedder.linear_2.', 'txt_in.c_embedder.fc_out.')
            new_key = new_key.replace('txt_in.individual_token_refiner.input_embedder.', 'txt_in.input_embedder.')
            new_key = new_key.replace('txt_in.individual_token_refiner.blocks.', 'txt_in.refiner_blocks.')

            # vector_in uses in_layer/out_layer -> fc_in/fc_out
            new_key = new_key.replace('vector_in.in_layer.', 'vector_in.fc_in.')
            new_key = new_key.replace('vector_in.out_layer.', 'vector_in.fc_out.')

            # final_layer adaLN uses indexed naming
            new_key = new_key.replace('final_layer.adaLN_modulation.1.', 'final_layer.adaLN_modulation.linear.')

            # Generic MLP naming for blocks (fc1/fc2 -> fc_in/fc_out)
            new_key = new_key.replace('.mlp.fc1.', '.mlp.fc_in.')
            new_key = new_key.replace('.mlp.fc2.', '.mlp.fc_out.')

            # img_mlp and txt_mlp in double_blocks
            new_key = new_key.replace('.img_mlp.fc1.', '.img_mlp.fc_in.')
            new_key = new_key.replace('.img_mlp.fc2.', '.img_mlp.fc_out.')
            new_key = new_key.replace('.txt_mlp.fc1.', '.txt_mlp.fc_in.')
            new_key = new_key.replace('.txt_mlp.fc2.', '.txt_mlp.fc_out.')

            # adaLN in refiner blocks
            new_key = new_key.replace('.adaLN_modulation.1.', '.adaLN_modulation.linear.')

            remapped[new_key] = value

        return remapped

    def load_weights(
        self,
        weights: dict[str, torch.Tensor],
        prefix: str = "",
        remap_keys: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Load weights with automatic key remapping.

        Args:
            weights: State dict to load
            prefix: Optional prefix to strip from keys
            remap_keys: Whether to remap keys from original format

        Returns:
            Tuple of (missing_keys, unexpected_keys)
        """
        if remap_keys:
            weights = self.remap_checkpoint_keys(weights)

        if prefix:
            weights = {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in weights.items()
            }

        missing, unexpected = self.load_state_dict(weights, strict=False)
        return missing, unexpected


# Entry point for model loading
EntryClass = HunyuanVideoAvatarTransformer
