#!/usr/bin/env python3
"""
Full end-to-end HunyuanVideo-Avatar video generation.

Generates a talking head video from:
- Reference portrait image
- Audio file
- Text prompt
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/home/zsi/projects/sglang/python")

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Paths
MODEL_BASE = "/home/zsi/.cache/cvlization/hunyuanvideo_avatar/weights/ckpts"
TRANSFORMER_PATH = f"{MODEL_BASE}/hunyuan-video-t2v-720p/transformers/"
VAE_PATH = f"{MODEL_BASE}/hunyuan-video-t2v-720p/vae/"
WHISPER_PATH = f"{MODEL_BASE}/whisper-tiny/"

# Sample inputs
SAMPLE_IMAGE = "/tmp/HunyuanVideo-Avatar/assets/image/1.png"
SAMPLE_AUDIO = "/tmp/HunyuanVideo-Avatar/assets/audio/2.WAV"
OUTPUT_PATH = "/tmp/avatar_output.mp4"


def init_distributed():
    """Initialize distributed environment."""
    from sglang.multimodal_gen.runtime.distributed import parallel_state
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from sglang.multimodal_gen.configs.pipeline_configs.hunyuan_avatar import HunyuanAvatarConfig

    server_args = ServerArgs(
        model_path="/tmp/hunyuan_avatar",
        num_gpus=1,
        host=None,
        port=None,
    )
    server_args.pipeline_config = HunyuanAvatarConfig()
    set_global_server_args(server_args)

    parallel_state.init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29502",
        local_rank=0,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    )
    parallel_state.initialize_model_parallel(
        data_parallel_size=1,
        classifier_free_guidance_degree=1,
        sequence_parallel_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    return server_args


def load_models():
    """Load all required models."""
    print("\n" + "=" * 60)
    print("Loading Models")
    print("=" * 60)

    # 1. Whisper
    print("\n[1/3] Loading Whisper...")
    from sglang.multimodal_gen.runtime.pipelines_core.stages import WhisperAudioEncoder
    whisper = WhisperAudioEncoder(WHISPER_PATH)
    whisper.to("cuda")
    print(f"  Whisper loaded")

    # 2. Transformer
    print("\n[2/3] Loading Avatar Transformer...")
    from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo_avatar import (
        HunyuanVideoAvatarTransformer,
        HunyuanVideoAvatarConfig,
    )
    config = HunyuanVideoAvatarConfig()
    transformer = HunyuanVideoAvatarTransformer(config, hf_config={})

    ckpt_file = list(Path(TRANSFORMER_PATH).glob("*_model_states.pt"))[0]
    state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if "module" in state_dict:
        state_dict = state_dict["module"]
    transformer.load_weights(state_dict, remap_keys=True)
    transformer = transformer.to("cuda", dtype=torch.bfloat16)
    transformer.eval()
    print(f"  Transformer loaded: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B params")

    # 3. VAE
    print("\n[3/3] Loading VAE...")
    from diffusers import AutoencoderKLHunyuanVideo
    vae = AutoencoderKLHunyuanVideo(
        in_channels=3,
        out_channels=3,
        latent_channels=16,
        down_block_types=(
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        up_block_types=(
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        norm_num_groups=32,
        scaling_factor=0.476986,
        temporal_compression_ratio=4,
        mid_block_add_attention=True,
    )
    weights_path = Path(VAE_PATH) / "pytorch_model.pt"
    vae.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
    vae = vae.to("cuda", dtype=torch.float16)
    vae.eval()
    print(f"  VAE loaded: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params")

    return whisper, transformer, vae


def encode_audio(whisper, audio_path, num_frames, fps=25.0):
    """Encode audio file."""
    print(f"\nEncoding audio: {audio_path}")
    audio_embeds = whisper.encode(audio_path, num_frames, fps)
    print(f"  Audio shape: {audio_embeds.shape}")
    return audio_embeds.to("cuda", dtype=torch.bfloat16)


def encode_reference_image(vae, image_path, height, width):
    """Encode reference image to latents."""
    print(f"\nEncoding reference image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(2)
    image_tensor = image_tensor * 2.0 - 1.0
    image_tensor = image_tensor.to("cuda", dtype=torch.float16)

    with torch.no_grad():
        vae.enable_tiling()
        latent_dist = vae.encode(image_tensor).latent_dist
        ref_latents = latent_dist.sample()
        if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
            ref_latents = (ref_latents - vae.config.shift_factor) * vae.config.scaling_factor
        else:
            ref_latents = ref_latents * vae.config.scaling_factor

    print(f"  Ref latents shape: {ref_latents.shape}")
    return ref_latents.to(dtype=torch.bfloat16)


def create_text_embeddings(batch_size=1, device="cuda", dtype=torch.bfloat16):
    """Create text embeddings (dummy for now - real would use LLaVA + CLIP)."""
    print("\nCreating text embeddings (dummy)...")
    encoder_hidden_states = torch.randn(batch_size, 256, 4096, device=device, dtype=dtype)
    pooled_embeds = torch.randn(batch_size, 768, device=device, dtype=dtype)
    return encoder_hidden_states, pooled_embeds


def run_denoising_loop(
    transformer,
    latents,
    encoder_hidden_states,
    pooled_embeds,
    ref_latents,
    audio_embeds,
    num_steps=30,
    guidance_scale=7.5,
    fps=24.0,
    use_batched_cfg=True,
):
    """Run the full denoising loop."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

    cfg_mode = "batched" if use_batched_cfg and guidance_scale > 1.0 else "sequential"
    print(f"\nRunning denoising loop ({num_steps} steps, CFG={guidance_scale}, mode={cfg_mode})...")

    # Create scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    scheduler.set_timesteps(num_steps, device=latents.device)
    timesteps = scheduler.timesteps

    batch_size = latents.shape[0]
    device = latents.device
    dtype = latents.dtype

    # FPS tensor
    fps_tensor = torch.tensor([fps], device=device, dtype=dtype).expand(batch_size)

    # Denoising loop
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        with torch.no_grad():
            with set_forward_context(current_timestep=i, attn_metadata=None):
                if use_batched_cfg and guidance_scale > 1.0:
                    # Batched CFG: run cond + uncond in single forward pass
                    latent_model_input = torch.cat([latents, latents])
                    timestep = t.expand(batch_size * 2)

                    # Concat conditioning (cond first, then uncond)
                    cond_hidden = torch.cat([
                        encoder_hidden_states,
                        torch.zeros_like(encoder_hidden_states),
                    ])
                    cond_pooled = torch.cat([
                        pooled_embeds,
                        torch.zeros_like(pooled_embeds),
                    ])
                    cond_ref = torch.cat([ref_latents, ref_latents])
                    cond_audio = torch.cat([audio_embeds, audio_embeds])
                    cond_fps = torch.cat([fps_tensor, fps_tensor])

                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=cond_hidden,
                        timestep=timestep,
                        ref_latents=cond_ref,
                        audio_embeds=cond_audio,
                        fps=cond_fps,
                        pooled_projections=cond_pooled,
                    )

                    # Split and apply CFG
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Sequential: run model twice (fallback)
                    timestep = t.expand(batch_size)
                    noise_pred_cond = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        ref_latents=ref_latents,
                        audio_embeds=audio_embeds,
                        fps=fps_tensor,
                        pooled_projections=pooled_embeds,
                    )

                    if guidance_scale > 1.0:
                        noise_pred_uncond = transformer(
                            hidden_states=latents,
                            encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                            timestep=timestep,
                            ref_latents=ref_latents,
                            audio_embeds=audio_embeds,
                            fps=fps_tensor,
                            pooled_projections=torch.zeros_like(pooled_embeds),
                        )
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


def decode_latents(vae, latents):
    """Decode latents to video frames."""
    print("\nDecoding latents to video...")

    with torch.no_grad():
        vae.enable_tiling()
        # Unscale
        if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor:
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        else:
            latents = latents / vae.config.scaling_factor

        video = vae.decode(latents.to(torch.float16)).sample

    # Convert to uint8
    video = (video + 1.0) / 2.0
    video = video.clamp(0, 1)
    video = (video * 255).to(torch.uint8)
    video = video[0].permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

    print(f"  Video shape: {video.shape}")
    return video


def save_video(video, output_path, fps=25):
    """Save video to file."""
    import imageio
    print(f"\nSaving video to: {output_path}")
    imageio.mimsave(output_path, video, fps=fps)
    print(f"  Saved {len(video)} frames at {fps} FPS")


def main():
    print("=" * 60)
    print("HunyuanVideo-Avatar Full Video Generation")
    print("=" * 60)

    # Check inputs
    if not os.path.exists(SAMPLE_IMAGE):
        print(f"Error: Image not found: {SAMPLE_IMAGE}")
        return 1
    if not os.path.exists(SAMPLE_AUDIO):
        print(f"Error: Audio not found: {SAMPLE_AUDIO}")
        return 1

    # Init
    init_distributed()

    # Config
    height, width = 512, 512  # Smaller for faster testing
    num_frames = 33  # ~1.3s at 25fps
    fps = 25.0
    num_steps = 30
    guidance_scale = 7.5

    latent_t = (num_frames - 1) // 4 + 1
    latent_h, latent_w = height // 8, width // 8
    latent_c = 16

    print(f"\nConfiguration:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames} (latent: {latent_t})")
    print(f"  Steps: {num_steps}, CFG: {guidance_scale}")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load models
    whisper, transformer, vae = load_models()

    # Encode inputs
    audio_embeds = encode_audio(whisper, SAMPLE_AUDIO, num_frames, fps)

    # Free Whisper memory
    del whisper
    torch.cuda.empty_cache()

    ref_latents = encode_reference_image(vae, SAMPLE_IMAGE, height, width)
    ref_latents = ref_latents.repeat(1, 1, latent_t, 1, 1)

    encoder_hidden_states, pooled_embeds = create_text_embeddings(device=device, dtype=dtype)

    # Create initial noise
    print("\nPreparing initial latents...")
    generator = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn(
        1, latent_c, latent_t, latent_h, latent_w,
        generator=generator, device=device, dtype=dtype
    )
    print(f"  Latents shape: {latents.shape}")

    # Run denoising
    latents = run_denoising_loop(
        transformer=transformer,
        latents=latents,
        encoder_hidden_states=encoder_hidden_states,
        pooled_embeds=pooled_embeds,
        ref_latents=ref_latents,
        audio_embeds=audio_embeds,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        fps=fps,
    )

    # Free transformer memory
    del transformer
    torch.cuda.empty_cache()

    # Decode
    video = decode_latents(vae, latents)

    # Save
    save_video(video, OUTPUT_PATH, fps=int(fps))

    print("\n" + "=" * 60)
    print("Video Generation Complete!")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
