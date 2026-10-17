"""TODO B20: symmetric offset attention bias for the GTrXL token mixer (CPU checks, no GPU needed).

Checks: the default `abs` mixer is unchanged (learned table, no offset table); the `sym` mixer has 66 offset values per head on the
6x6 grid, its attention bias depends only on the token offset (translation) and is mirror-invariant in x, and at init the mixer is
still the identity (zero output projection), like the abs mixer.
"""
import torch

from atari_qwen.models.ez_model import TokenMixer


def _grid_flip(idx: int, s: int) -> int:
    r, c = divmod(idx, s)
    return r * s + (s - 1 - c)


def test_abs_mixer_unchanged():
    m = TokenMixer(channels=64, hw=36, dim=128, depth=2, heads=4, pos_mode="abs")
    assert hasattr(m, "pos") and m.pos.shape == (1, 36, 128)
    assert not hasattr(m, "rel_table")
    x = torch.randn(3, 64, 6, 6)
    assert m(x).shape == x.shape


def test_sym_mixer_has_66_offsets_per_head():
    m = TokenMixer(channels=64, hw=36, dim=128, depth=2, heads=4, pos_mode="sym")
    assert not hasattr(m, "pos")
    assert tuple(m.rel_table.shape) == (4, 66)
    x = torch.randn(3, 64, 6, 6)
    assert m(x).shape == x.shape


def test_sym_bias_depends_only_on_offset():
    m = TokenMixer(channels=64, hw=36, dim=128, depth=2, heads=4, pos_mode="sym")
    with torch.no_grad():
        m.rel_table.normal_()                       # arbitrary values: the property must hold for any table
        bias = m.rel_table[:, m.rel_idx]            # (heads, 36, 36)
    s = 6
    # token (r, c) -> index r * s + c. Pair (1,1)->(2,3) has dx = 2, dy = -1; the pair (2,2)->(3,4) has the same offset.
    a, b = 1 * s + 1, 2 * s + 3
    a2, b2 = 2 * s + 2, 3 * s + 4
    assert torch.allclose(bias[:, a, b], bias[:, a2, b2])


def test_sym_bias_is_mirror_invariant():
    m = TokenMixer(channels=64, hw=36, dim=128, depth=2, heads=4, pos_mode="sym")
    with torch.no_grad():
        m.rel_table.normal_()
        bias = m.rel_table[:, m.rel_idx]
    s = 6
    for a in range(36):
        for b in range(36):
            assert torch.allclose(bias[:, a, b], bias[:, _grid_flip(a, s), _grid_flip(b, s)])


def test_sym_mixer_is_identity_at_init():
    torch.manual_seed(0)
    m = TokenMixer(channels=64, hw=36, dim=128, depth=2, heads=4, pos_mode="sym").eval()
    x = torch.randn(2, 64, 6, 6)
    with torch.no_grad():
        assert torch.allclose(m(x), x, atol=1e-6)
