"""TODO B70b: --mixer-no-pos drops the learned position table from the policy-token trunk's tokens; the default keeps it."""
import torch

from atari_qwen.models.ez_model import PolicyTokenTrunk


def _trunk(no_pos):
    torch.manual_seed(0)
    return PolicyTokenTrunk(channels=8, hw=36, dim=16, depth=1, heads=2, ffn_dim=32, num_actions=4, support_size=5, no_pos=no_pos)


def test_no_pos_drops_the_position_term():
    t = _trunk(True)
    s = torch.randn(2, 8, 6, 6)
    with torch.no_grad():
        base = t.inp(s.flatten(2).transpose(1, 2))
        assert torch.allclose(t.tokens(s)[:, 1:], base)
        t.pos.data.fill_(3.0)
        assert torch.allclose(t.tokens(s)[:, 1:], base)


def test_default_keeps_the_position_term():
    t = _trunk(False)
    s = torch.randn(2, 8, 6, 6)
    with torch.no_grad():
        base = t.inp(s.flatten(2).transpose(1, 2))
        t.pos.data.fill_(0.0)
        assert torch.allclose(t.tokens(s)[:, 1:], base)
        t.pos.data.fill_(3.0)
        assert not torch.allclose(t.tokens(s)[:, 1:], base)
