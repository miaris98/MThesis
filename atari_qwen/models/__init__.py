"""Model components for Qwen Atari agent."""
from atari_qwen.models.visual_encoders import (
    NatureCNNEncoder,
    ImpalaCNNEncoder,
    PatchTokenizer
)
try:
    from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic
    __all__ = [
        "NatureCNNEncoder",
        "ImpalaCNNEncoder",
        "PatchTokenizer",
        "QwenAtariActorCritic"
    ]
except ImportError:
    __all__ = [
        "NatureCNNEncoder",
        "ImpalaCNNEncoder",
        "PatchTokenizer"
    ]
