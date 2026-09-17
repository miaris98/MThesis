"""Unit tests for Qwen Atari Actor-Critic policy network and components."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:
    pytest = None
import torch

from src.models.transformer.layers import RMSNorm, SwiGLU, QwenAttentionWithSkip, QwenTransformerBlock
from atari_qwen.models.visual_encoders import NatureCNNEncoder, ImpalaCNNEncoder, PatchTokenizer
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic
from atari_qwen.config.atari_config import QWEN_ATARI_PRESETS, get_config


def test_transformer_layer_imports():
    """Verify core Qwen building blocks imported from src.models.transformer."""
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 10, 64)
    out = norm(x)
    assert out.shape == (2, 10, 64)

    swiglu = SwiGLU(dim=64, hidden_dim=128)
    out = swiglu(x)
    assert out.shape == (2, 10, 64)

    block = QwenTransformerBlock(dim=64, num_heads=4, ffn_dim=128)
    out = block(x)
    assert out.shape == (2, 10, 64)
    assert block.alpha_attn.shape == (64,)
    assert block.alpha_ffn.shape == (64,)


def test_visual_encoders():
    """Verify visual tokenizers produce expected sequence shapes for 84x84 frame stacks."""
    batch_size = 4
    x = torch.randn(batch_size, 4, 84, 84)

    # 1. Nature CNN
    nature = NatureCNNEncoder(in_channels=4, embed_dim=128, return_spatial_tokens=True)
    out_nature = nature(x)
    assert out_nature.shape == (batch_size, 49, 128)

    # 2. Impala CNN
    impala = ImpalaCNNEncoder(in_channels=4, embed_dim=128, return_spatial_tokens=True)
    out_impala = impala(x)
    assert out_impala.shape == (batch_size, 121, 128)

    # 3. Patch Tokenizer
    patch = PatchTokenizer(in_channels=4, embed_dim=128, patch_size=14)
    out_patch = patch(x)
    assert out_patch.shape == (batch_size, 36, 128)


def test_qwen_atari_actor_critic_forward_and_backward():
    """Verify full forward, sampling, and backward gradient flow through Qwen transformer."""
    batch_size = 2
    action_dim = 6  # e.g. Pong action space
    x = torch.randn(batch_size, 4, 84, 84)

    model = QwenAtariActorCritic(
        action_dim=action_dim,
        in_channels=4,
        preset="tiny",
        encoder_type="nature_cnn"
    )

    # Test get_action_and_value
    action, log_prob, entropy, value = model.get_action_and_value(x)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert value.shape == (batch_size,)

    # Test value estimation
    v = model.get_value(x)
    assert v.shape == (batch_size,)

    # Test backward pass
    loss = -log_prob.mean() + value.mean() - 0.01 * entropy.mean()
    loss.backward()

    # Verify gradients flowed into Qwen transformer blocks and alpha gates
    first_block = model.blocks[0]
    assert first_block.alpha_attn.grad is not None
    assert first_block.alpha_ffn.grad is not None
    assert model.actor_token.grad is not None
    assert model.critic_token.grad is not None


def test_qwen_atari_presets():
    """Verify preset dimensions match configurations."""
    for preset_name in ["tiny", "small", "100m"]:
        model = QwenAtariActorCritic(
            action_dim=4,
            preset=preset_name
        )
        expected = QWEN_ATARI_PRESETS[preset_name]
        assert model.depth == expected["depth"]
        assert model.embed_dim == expected["embed_dim"]
        assert model.num_heads == expected["num_heads"]


if __name__ == "__main__":
    test_transformer_layer_imports()
    test_visual_encoders()
    test_qwen_atari_actor_critic_forward_and_backward()
    test_qwen_atari_presets()
    print("✓ All Qwen Atari tests passed successfully!")
