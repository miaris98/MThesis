"""TODO B61: Qwen blocks as the EZ-V2 token mixer (`trunk="qwen"`). Same interface as the GTrXL mixer, near identity at
init, and gradient reaches the blocks from the first update when the output projection is not zero-initialised (S-116)."""
import pytest
import torch

from atari_qwen.models.ez_model import MIXER_SIZES, DiscreteSupport, EZV2Model


def _model(trunk="qwen", size="small", std=0.02):
    torch.manual_seed(0)
    return EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk=trunk, state_hw=6, mixer_size=size,
                     mixer_out_init_std=std)


@pytest.mark.parametrize("size", ["small", "30m"])
def test_shapes_match_the_resnet_trunk(size):
    m, ref = _model(size=size).eval(), _model("resnet", None).eval()
    x = torch.rand(2, 12, 96, 96)
    with torch.no_grad():
        s, v, p = m.initial_inference(x)
        s0, v0, p0 = ref.initial_inference(x)
        assert s.shape == s0.shape == (2, 64, 6, 6) and v.shape == v0.shape and p.shape == p0.shape
        assert m.dynamics(s, torch.tensor([1, 3])).shape == s.shape


def test_30m_mixer_has_the_qwen30m_shape():
    dim, depth, heads, ffn = MIXER_SIZES["30m"]
    mix = _model(size="30m").repr_mixer
    assert (dim, depth, heads, ffn) == (512, 8, 8, 1536) and len(mix.blocks) == depth
    assert mix.blocks[0].attn.num_heads == heads and mix.blocks[0].ffn.w_up.out_features == ffn
    n = sum(q.numel() for q in mix.blocks.parameters())
    assert 26e6 < n < 29e6  # 8 x (4 x 512^2 + 3 x 512 x 1536) + norms and gates


def test_zero_init_output_is_exact_identity_and_small_init_is_near_identity():
    s = torch.randn(2, 64, 6, 6)
    assert torch.equal(_model(std=0.0).repr_mixer(s), s)
    out = _model(std=0.02).repr_mixer(s)
    assert not torch.equal(out, s)
    assert float((out - s).norm() / s.norm()) < 0.5


@pytest.mark.parametrize("size", ["small", "30m"])
def test_gradient_reaches_every_block(size):
    m = _model(size=size, std=0.02)
    m.repr_mixer(torch.randn(2, 64, 6, 6)).square().mean().backward()
    for blk in m.repr_mixer.blocks:
        assert float(blk.attn.q_proj.weight.grad.abs().max()) > 0.0
        assert float(blk.ffn.w_gate.weight.grad.abs().max()) > 0.0


def test_mixer_parameters_are_found_by_the_trainers_name_filter():
    m = _model()
    names = [n for n, _ in m.named_parameters() if "mixer" in n]
    assert any("alpha_attn" in n for n in names) and any("q_proj" in n for n in names)
    rest = [n for n, _ in m.named_parameters() if "mixer" not in n]
    assert not any("blocks" in n for n in rest)


def test_qwen_rejects_the_offset_bias():
    with pytest.raises(AssertionError):
        EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk="qwen", state_hw=6, mixer_pos="sym")
