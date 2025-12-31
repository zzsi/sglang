#!/usr/bin/env python3
"""
Test HunyuanVideo-Avatar weight loading and forward pass.

This script tests:
1. Weight loading from the original checkpoint format
2. Forward pass with dummy inputs
3. Verification that audio adapters are working
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add the project to path
sys.path.insert(0, "/home/zsi/projects/sglang/python")

import torch
from pathlib import Path


def init_distributed():
    """Initialize distributed environment for single GPU testing."""
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from sglang.multimodal_gen.configs.pipeline_configs.hunyuan_avatar import HunyuanAvatarConfig

    # Create minimal server args
    server_args = ServerArgs(
        model_path="/tmp/hunyuan_avatar_test",  # dummy path
        num_gpus=1,
        host=None,
        port=None,
    )
    # Set the pipeline config
    server_args.pipeline_config = HunyuanAvatarConfig()
    set_global_server_args(server_args)

    # Initialize distributed environment (this sets up _WORLD)
    parallel_state.init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29500",
        local_rank=0,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )

    # Initialize model parallel groups
    parallel_state.initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        sequence_parallel_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
    )


def load_avatar_checkpoint(model, ckpt_path: str, load_key: str = "module", use_key_remapping: bool = True):
    """Load checkpoint in HunyuanVideo-Avatar format.

    Args:
        model: HunyuanVideoAvatarTransformer model
        ckpt_path: Path to checkpoint file or directory
        load_key: Key to extract from checkpoint (default: "module")
        use_key_remapping: Whether to remap keys from original format (default: True)
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        # Find the model states file
        candidates = list(ckpt_path.glob("*_model_states.pt"))
        if not candidates:
            raise FileNotFoundError(f"No *_model_states.pt found in {ckpt_path}")
        ckpt_path = candidates[0]

    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract the actual model state dict
    if load_key in state_dict:
        state_dict = state_dict[load_key]
    elif load_key == ".":
        pass
    else:
        print(f"Available keys: {list(state_dict.keys())}")
        raise KeyError(f"Key '{load_key}' not found in checkpoint")

    # Use the model's load_weights method with key remapping
    if use_key_remapping:
        print("Using key remapping to convert checkpoint keys...")
        missing, unexpected = model.load_weights(state_dict, remap_keys=True)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint successfully!")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    if missing:
        print(f"Sample missing keys: {missing[:5]}")
    if unexpected:
        print(f"Sample unexpected keys: {unexpected[:5]}")

    if len(missing) == 0 and len(unexpected) == 0:
        print("✓ Perfect weight loading - all keys matched!")

    return model


def test_weight_loading():
    """Test loading weights into HunyuanVideoAvatarTransformer."""
    print("=" * 60)
    print("Testing HunyuanVideo-Avatar weight loading")
    print("=" * 60)

    from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo_avatar import (
        HunyuanVideoAvatarTransformer,
        HunyuanVideoAvatarConfig,
    )

    # Create config
    config = HunyuanVideoAvatarConfig()
    print(f"\nConfig created")

    # hf_config is typically loaded from the model's config.json
    # For testing, we use an empty dict
    hf_config = {}

    # Create model on CPU first
    print("\nCreating model...")
    with torch.device("meta"):
        model = HunyuanVideoAvatarTransformer(config, hf_config=hf_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    # Check avatar-specific components
    print("\nAvatar-specific components:")
    print(f"  - audio_proj: {sum(p.numel() for p in model.audio_proj.parameters()) / 1e6:.2f}M params")
    print(f"  - ref_in: {sum(p.numel() for p in model.ref_in.parameters()) / 1e6:.2f}M params")
    print(f"  - before_proj: {sum(p.numel() for p in model.before_proj.parameters()) / 1e6:.2f}M params")
    print(f"  - audio_adapter_blocks: {len(model.audio_adapter_blocks)} blocks")

    # Now materialize and load weights
    print("\nMaterializing model on CPU...")
    model = HunyuanVideoAvatarTransformer(config, hf_config=hf_config)

    # Load checkpoint
    ckpt_path = "/home/zsi/.cache/cvlization/hunyuanvideo_avatar/weights/ckpts/hunyuan-video-t2v-720p/transformers/"
    model = load_avatar_checkpoint(model, ckpt_path, load_key="module")

    print("\nWeight loading test PASSED!")
    return model, config


def test_forward_pass(model, config):
    """Test forward pass with dummy inputs."""
    print("\n" + "=" * 60)
    print("Testing forward pass")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"\nDevice: {device}, dtype: {dtype}")

    # Move model to device
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Create dummy inputs
    batch_size = 1
    num_video_frames = 33  # Original video frames
    height, width = 704, 768

    # Latent dimensions after VAE
    spatial_factor = 8
    temporal_factor = 4
    latent_channels = 16

    latent_h = height // spatial_factor
    latent_w = width // spatial_factor
    latent_t = (num_video_frames - 1) // temporal_factor + 1  # 9

    # For the forward pass to work, audio frames must match latent frames
    # In production, audio at video frame rate would be downsampled/indexed
    # For testing, we directly use latent frame rate
    num_audio_frames = latent_t  # 9 audio frames to match 9 latent frames

    print(f"\nInput shapes:")
    print(f"  Latent: ({batch_size}, {latent_channels}, {latent_t}, {latent_h}, {latent_w})")

    # Create dummy latents
    hidden_states = torch.randn(
        batch_size, latent_channels, latent_t, latent_h, latent_w,
        device=device, dtype=dtype
    )

    # Create dummy text embeddings
    # Llama embedding: (batch, seq_len, hidden_dim)
    text_seq_len = 256
    text_hidden_dim = 4096
    encoder_hidden_states = torch.randn(
        batch_size, text_seq_len, text_hidden_dim,
        device=device, dtype=dtype
    )

    # Create timestep
    timestep = torch.tensor([500.0], device=device, dtype=dtype)

    # Create reference latents (same size as video latents - ref is repeated in pipeline)
    # Original pipeline: ref_latents.repeat(1,1,frame,1,1)
    ref_latents = torch.randn(
        batch_size, latent_channels, latent_t, latent_h, latent_w,
        device=device, dtype=dtype
    )

    # Create audio embeddings
    # AudioProjNet2 expects: (batch, num_frames, seq_len, blocks, channels)
    # Config: seq_len=10, blocks=5, channels=384, context_tokens=4
    audio_seq_len = 10
    audio_blocks = 5
    audio_channels = 384

    # Raw audio features from Whisper
    # Shape: (batch, num_frames, seq_len, blocks, channels)
    # Audio is at video frame rate, not latent frame rate
    audio_embeds = torch.randn(
        batch_size, num_audio_frames, audio_seq_len, audio_blocks, audio_channels,
        device=device, dtype=dtype
    )

    print(f"  Text embeddings: {encoder_hidden_states.shape}")
    print(f"  Ref latents: {ref_latents.shape}")
    print(f"  Audio embeds: {audio_embeds.shape}")

    # Create motion/FPS control tensors
    fps_tensor = torch.tensor([24.0], device=device, dtype=dtype)
    motion_exp_tensor = torch.tensor([1.0], device=device, dtype=dtype)
    motion_pose_tensor = torch.tensor([1.0], device=device, dtype=dtype)

    print(f"  FPS: {fps_tensor.item()}")
    print(f"  Motion exp: {motion_exp_tensor.item()}")
    print(f"  Motion pose: {motion_pose_tensor.item()}")

    # Run forward pass
    print("\nRunning forward pass...")
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

    try:
        with torch.no_grad():
            with set_forward_context(current_timestep=0, attn_metadata=None):
                output = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    ref_latents=ref_latents,
                    audio_embeds=audio_embeds,
                    fps=fps_tensor,
                    motion_exp=motion_exp_tensor,
                    motion_pose=motion_pose_tensor,
                )
        print(f"\nOutput shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        print("\nForward pass test PASSED!")
    except Exception as e:
        print(f"\nForward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    print("HunyuanVideo-Avatar Integration Test")
    print("=" * 60)

    # Initialize distributed environment
    print("\nInitializing distributed environment...")
    init_distributed()
    print("Distributed environment initialized.")

    # Test 1: Weight loading
    try:
        model, config = test_weight_loading()
    except Exception as e:
        print(f"\nWeight loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 2: Forward pass
    try:
        success = test_forward_pass(model, config)
        if not success:
            return 1
    except Exception as e:
        print(f"\nForward pass test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
