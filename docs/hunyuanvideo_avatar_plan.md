# HunyuanVideo-Avatar Integration Plan for SGLang

## Overview

This document outlines the plan to integrate HunyuanVideo-Avatar (Tencent's audio-driven avatar generation) into SGLang's multimodal generation framework.

## Current State

### SGLang HunyuanVideo Support
- Base HunyuanVideo DiT: ✅ Working
- Pipeline: `HunyuanVideoPipeline`
- Model: `HunyuanVideoTransformer` (MMDoubleStreamBlock + MMSingleStreamBlock)
- Location: `python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py`

### HunyuanVideo-Avatar Components (from `/tmp/HunyuanVideo-Avatar`)
The avatar version adds:
1. **AudioProjNet2** - Projects audio embeddings to context tokens
2. **PerceiverAttentionCA** - Cross-attention for audio conditioning
3. **HYVideoDiffusionTransformer** - Modified DiT with audio injection points
4. **HunyuanVideoAudioPipeline** - Extended pipeline with audio processing

---

## Architecture Differences

### Base HunyuanVideo (SGLang)
```
Text Encoder → Text Embeddings
                    ↓
Timestep → Modulation
                    ↓
[MMDoubleStreamBlock × 20] → [MMSingleStreamBlock × 40] → Output
```

### HunyuanVideo-Avatar
```
Text Encoder → Text Embeddings ──────────────────────────────┐
                                                              │
Audio (Whisper) → AudioProjNet2 → Audio Context Tokens ──────┤
                                                              │
Reference Image → CLIP → Image Embeddings ───────────────────┤
                                                              ↓
Timestep → Modulation
                    ↓
[DoubleStreamBlock × 20] ←── PerceiverAttentionCA (audio injection)
            ↓
[SingleStreamBlock × 40]
            ↓
         Output
```

---

## Implementation Plan

### Phase 1: Audio Adapter Modules (2-3 files)

#### 1.1 Add AudioProjNet2
**File:** `python/sglang/multimodal_gen/runtime/models/adapters/audio_proj.py`

```python
class AudioProjNet2(nn.Module):
    """Projects audio embeddings to context tokens for DiT conditioning."""

    def __init__(
        self,
        seq_len: int = 5,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 4,
    ):
        # 3-layer MLP: input_dim → intermediate_dim → context_tokens * output_dim
        pass

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        # Input: (batch, frames, window, blocks, channels)
        # Output: (batch, frames, context_tokens, output_dim)
        pass
```

#### 1.2 Add PerceiverAttentionCA
**File:** `python/sglang/multimodal_gen/runtime/models/adapters/perceiver_ca.py`

```python
class PerceiverAttentionCA(nn.Module):
    """Cross-attention for injecting audio features into visual latents."""

    def __init__(self, dim: int = 3072, dim_head: int = 1024, heads: int = 33):
        # Q from latents, K/V from audio
        pass

    def forward(self, audio_features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        # Cross-attend audio to visual latents
        pass
```

---

### Phase 2: Modified DiT Model (1-2 files)

#### 2.1 Create HunyuanVideoAvatarTransformer
**File:** `python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo_avatar.py`

Key modifications from base HunyuanVideo:

```python
class HunyuanVideoAvatarTransformer(CachableDiT, OffloadableDiTMixin):
    """HunyuanVideo with audio conditioning for avatar generation."""

    def __init__(self, config: HunyuanVideoAvatarConfig):
        super().__init__()
        # Base transformer blocks (same as HunyuanVideo)
        self.double_blocks = nn.ModuleList([...])
        self.single_blocks = nn.ModuleList([...])

        # NEW: Audio conditioning components
        self.audio_proj = AudioProjNet2(
            seq_len=config.audio_seq_len,
            blocks=config.audio_blocks,
            channels=config.audio_channels,
            output_dim=config.hidden_size,
            context_tokens=config.audio_context_tokens,
        )

        # Audio injection at specific layers
        self.audio_injection_layers = config.audio_injection_layers  # e.g., [0, 4, 8, 12, 16]
        self.audio_cross_attn = nn.ModuleDict({
            str(i): PerceiverAttentionCA(dim=config.hidden_size)
            for i in self.audio_injection_layers
        })

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_embeds: torch.Tensor | None = None,  # NEW
        **kwargs,
    ) -> torch.Tensor:
        # Process audio embeddings
        if audio_embeds is not None:
            audio_context = self.audio_proj(audio_embeds)

        # Double stream blocks with audio injection
        for i, block in enumerate(self.double_blocks):
            hidden_states = block(hidden_states, ...)

            # Inject audio at specified layers
            if audio_embeds is not None and i in self.audio_injection_layers:
                audio_delta = self.audio_cross_attn[str(i)](audio_context, hidden_states)
                hidden_states = hidden_states + audio_delta

        # Single stream blocks (no audio injection)
        for block in self.single_blocks:
            hidden_states = block(hidden_states, ...)

        return hidden_states
```

---

### Phase 3: Config & Registry (2-3 files)

#### 3.1 Add Config Classes
**File:** `python/sglang/multimodal_gen/configs/models/dits/hunyuanvideo_avatar.py`

```python
@dataclass
class HunyuanVideoAvatarArchConfig:
    # Base HunyuanVideo params
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 20
    num_single_layers: int = 40

    # Audio adapter params
    audio_seq_len: int = 5
    audio_blocks: int = 12
    audio_channels: int = 768
    audio_context_tokens: int = 4
    audio_injection_layers: list[int] = field(
        default_factory=lambda: [0, 4, 8, 12, 16]
    )
```

#### 3.2 Add Sampling Params
**File:** `python/sglang/multimodal_gen/configs/sample/hunyuan_avatar.py`

```python
@dataclass
class HunyuanAvatarSamplingParams(SamplingParams):
    # Video params
    num_frames: int = 129
    height: int = 704
    width: int = 768
    fps: int = 24

    # Audio params
    audio_path: str | None = None
    audio_embeds: np.ndarray | None = None  # Pre-computed whisper features
```

---

### Phase 4: Pipeline (1 file)

#### 4.1 Create HunyuanVideoAvatarPipeline
**File:** `python/sglang/multimodal_gen/runtime/pipelines/hunyuan_avatar_pipeline.py`

```python
class HunyuanVideoAvatarPipeline(ComposedPipelineBase):
    """Pipeline for audio-driven avatar generation."""

    pipeline_name = "hunyuan_avatar"

    def __init__(self, ...):
        super().__init__(...)
        # Additional: Whisper audio encoder (optional, for raw audio input)
        self.audio_encoder = None  # Load if needed

    def prepare_audio_embeds(
        self,
        audio_path: str | None,
        audio_embeds: np.ndarray | None,
        num_frames: int,
    ) -> torch.Tensor:
        """Prepare audio embeddings from path or pre-computed."""
        if audio_embeds is not None:
            return torch.from_numpy(audio_embeds)
        elif audio_path is not None:
            # Load and encode with Whisper
            return self._encode_audio(audio_path, num_frames)
        return None

    def _encode_audio(self, audio_path: str, num_frames: int) -> torch.Tensor:
        """Encode audio file using Whisper."""
        # Implementation using whisper model
        pass
```

---

### Phase 5: Weight Loading & Registry (2 files)

#### 5.1 Add Weight Loader
**File:** Extend `python/sglang/multimodal_gen/runtime/loader/component_loader.py`

```python
def load_hunyuan_avatar_transformer(
    model_path: str,
    config: HunyuanVideoAvatarConfig,
    ...
) -> HunyuanVideoAvatarTransformer:
    """Load transformer with audio adapter weights."""
    # Load base transformer weights
    # Load audio_proj weights
    # Load audio_cross_attn weights
    pass
```

#### 5.2 Register Model
**File:** Extend `python/sglang/multimodal_gen/registry.py`

```python
register_configs(
    sampling_param_cls=HunyuanAvatarSamplingParams,
    pipeline_config_cls=HunyuanAvatarPipelineConfig,
    hf_model_paths=[
        "tencent/HunyuanVideo-Avatar",
    ],
    model_detectors=[lambda hf_id: "hunyuan" in hf_id.lower() and "avatar" in hf_id.lower()],
)
```

---

## File Summary

| Phase | File | Description |
|-------|------|-------------|
| 1 | `runtime/models/adapters/audio_proj.py` | AudioProjNet2 module |
| 1 | `runtime/models/adapters/perceiver_ca.py` | PerceiverAttentionCA module |
| 2 | `runtime/models/dits/hunyuanvideo_avatar.py` | Modified DiT with audio injection |
| 3 | `configs/models/dits/hunyuanvideo_avatar.py` | Architecture config |
| 3 | `configs/sample/hunyuan_avatar.py` | Sampling params with audio |
| 3 | `configs/pipeline_configs/hunyuan_avatar.py` | Pipeline config |
| 4 | `runtime/pipelines/hunyuan_avatar_pipeline.py` | Avatar generation pipeline |
| 5 | `runtime/loader/component_loader.py` | Weight loading (extend) |
| 5 | `registry.py` | Model registration (extend) |

---

## Dependencies

### Required
- Existing SGLang HunyuanVideo support (✅ verified working)
- HunyuanVideo-Avatar weights (from Tencent)

### Optional (for raw audio input)
- `whisper` - For audio encoding
- `librosa` or `torchaudio` - For audio preprocessing

---

## Testing Plan

1. **Unit Tests**
   - AudioProjNet2 forward pass
   - PerceiverAttentionCA attention computation
   - Weight loading verification

2. **Integration Tests**
   - Full pipeline with pre-computed audio embeddings
   - End-to-end with raw audio file

3. **Benchmark**
   - VRAM usage comparison (base vs avatar)
   - Latency impact of audio conditioning

---

## Estimated Effort

| Phase | Estimated Time | Priority |
|-------|---------------|----------|
| Phase 1: Audio Adapters | 1-2 days | High |
| Phase 2: Modified DiT | 2-3 days | High |
| Phase 3: Configs | 0.5 day | High |
| Phase 4: Pipeline | 1-2 days | High |
| Phase 5: Loading & Registry | 1 day | High |
| Testing & Debug | 2-3 days | Medium |

**Total: ~8-12 days**

---

## Future Enhancements

1. **Multi-character support** - FAA (Face-Aware Audio Adapter)
2. **Emotion control** - AEM (Audio Emotion Module)
3. **Real-time streaming** - Chunk-based generation
4. **LoRA support** - Fine-tuning with custom avatars
