# SPDX-License-Identifier: Apache-2.0
"""
Audio and multimodal adapters for avatar generation models.
"""

from sglang.multimodal_gen.runtime.models.adapters.audio_proj import AudioProjNet2
from sglang.multimodal_gen.runtime.models.adapters.perceiver_ca import PerceiverAttentionCA

__all__ = [
    "AudioProjNet2",
    "PerceiverAttentionCA",
]
