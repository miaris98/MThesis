"""TODO (2026-11-03): determinism. Two runs from the same seed must start from identical weights and take an identical first step. Without
this, Gate screens are noisy and two runs that differ only in a flag cannot be compared; a silent source of randomness shows up here first."""
import torch
import torch.nn.functional as F

from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport


def _model(seed):
    torch.manual_seed(seed)
    return EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk="conv_transformer", norm="batch", state_hw=6,
                     mixer_out_init_std=0.0, mixer_bg_init=0.0, simnorm=0, mixer_pos="rope", mixer_size="mini",
                     mixer_qk_norm=False, mixer_where="pred", mixer_no_pos=False)


def test_the_same_seed_gives_identical_weights():
    a, b = _model(3).state_dict(), _model(3).state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_the_same_seed_gives_an_identical_first_step():
    x = torch.rand(8, 12, 96, 96, generator=torch.Generator().manual_seed(5))
    results = []
    for _ in range(2):
        m = _model(3).train()
        s, _, _ = m.initial_inference(x)
        _, logits = m.pred_mixer(s)
        loss = F.cross_entropy(logits, torch.zeros(8, dtype=torch.long))
        loss.backward()
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        opt.step()
        results.append((float(loss), {k: v.clone() for k, v in m.state_dict().items() if v.dtype.is_floating_point}))
    assert results[0][0] == results[1][0]
    assert all(torch.equal(results[0][1][k], results[1][1][k]) for k in results[0][1])
