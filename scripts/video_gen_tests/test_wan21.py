#!/usr/bin/env python3
"""Test WAN 2.1 T2V 1.3B with SGLang diffusion."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sglang.multimodal_gen import DiffGenerator

def main():
    print("Loading Wan2.1-T2V-1.3B model...")

    # Set host=None and port=None to enable local mode (auto-start server)
    generator = DiffGenerator.from_pretrained(
        model_path="/tmp/wan21_model",  # WAN 2.1 1.3B
        num_gpus=1,
        host=None,  # Enable local mode
        port=None,  # Enable local mode
    )

    print("Generating video...")
    video = generator.generate(
        sampling_params_kwargs=dict(
            prompt="A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest.",
            num_frames=41,  # Shorter for faster test
            num_inference_steps=20,  # Fewer steps for faster test
            height=480,
            width=832,
            output_path="/tmp/sglang_test_output/",
            save_output=True,
        )
    )

    print(f"Video generation complete!")
    print(f"Output: {video}")

    # Cleanup
    generator.shutdown()

if __name__ == "__main__":
    main()
