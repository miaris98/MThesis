"""GTrXL token mixer init (TODO B36, S-116): the original zero-init output projection is the default (exact identity at init);
mixer_out_init_std > 0 gives gradient to the attention/FFN blocks from the first update."""
import torch

from atari_qwen.models.ez_model import DiscreteSupport, EZV2Model


def _model(std):
    torch.manual_seed(0)
    return EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk="gtrxl", state_hw=6, mixer_out_init_std=std)


def test_default_is_zero_init_identity():
    m = _model(0.0)
    assert float(m.repr_mixer.out.weight.abs().max()) == 0.0 and float(m.dyn_mixer.out.bias.abs().max()) == 0.0
    s = torch.randn(2, 64, 6, 6)
    assert torch.equal(m.repr_mixer(s), s)


def test_out_init_std_opens_the_mixer_and_lets_gradient_reach_the_blocks():
    m = _model(0.02)
    assert 0.01 < float(m.repr_mixer.out.weight.std()) < 0.03
    s = torch.randn(2, 64, 6, 6)
    assert not torch.equal(m.repr_mixer(s), s)
    m.repr_mixer(s).square().mean().backward()
    assert float(m.repr_mixer.blocks[0].q_proj.weight.grad.abs().max()) > 0.0
    zero = _model(0.0)
    zero.repr_mixer(s).square().mean().backward()
    # zero output projection: the blocks upstream of it get exactly zero gradient at init
    assert float(zero.repr_mixer.blocks[0].v_proj.weight.grad.abs().max()) == 0.0
