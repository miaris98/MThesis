"""TODO B68 D1: --mixer-where repr keeps the Qwen mixer in the representation only, and the dynamics is EZ-V2's conv + ResNet alone.

"both" (the default) must be the existing model exactly, so every checkpoint and running arm is untouched. "repr" must drop the dynamics
mixer's parameters, compute the dynamics as the ResNet path written out by hand, still mix the representation, and round-trip through a
checkpoint built from its stamped args (the deferred evals rebuild the model that way)."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from atari_qwen.models.ez_model import EZV2Model
from atari_qwen.training.train_ez_offpolicy import mixer_param_groups, parse_args


def _model(where, seed=0):
    torch.manual_seed(seed)
    m = EZV2Model(4, obs_channels=12, trunk="qwen", mixer_size="small", mixer_qk_norm=True, mixer_where=where)
    torch.nn.init.normal_(m.repr_mixer.out.weight, std=0.02)   # open the zero-initialised output so the mixer changes the state
    return m.eval()


def test_default_is_both_and_unchanged():
    assert parse_args([]).mixer_where == "both"
    keys = set(EZV2Model(4, obs_channels=12, trunk="qwen", mixer_size="small", mixer_qk_norm=True).state_dict())
    assert keys == set(_model("both").state_dict())
    assert any(k.startswith("dyn_mixer.") for k in keys)


def test_repr_drops_the_dynamics_mixer():
    both, repr_only = _model("both"), _model("repr")
    assert repr_only.dyn_mixer is None
    kb, kr = set(both.state_dict()), set(repr_only.state_dict())
    assert kr == {k for k in kb if not k.startswith("dyn_mixer.")}
    names = {n for n, _ in repr_only.named_parameters()}
    groups = mixer_param_groups(repr_only, 1e-4)
    assert sum(len(g["params"]) for g in groups) == sum(1 for n in names if "mixer" in n) > 0


def test_repr_dynamics_is_the_resnet_path_and_representation_still_mixes():
    m = _model("repr")
    obs, a = torch.rand(3, 12, 96, 96), torch.tensor([0, 2, 3])
    with torch.no_grad():
        pre = m.repr_res(m.down(obs))
        s = m.representation(obs)
        assert not torch.allclose(s, pre)                         # the Qwen mixer is still in the representation
        plane = (a.float() / m.action_dim).view(-1, 1, 1, 1).expand(-1, 1, m.hw, m.hw)
        plane = F.relu(m.act_ln(m.act_conv(plane)))
        ref = m.dyn_res(F.relu(m.dyn_bn(m.dyn_conv(torch.cat([s, plane], 1))) + s))
        assert torch.equal(m.dynamics(s, a), ref)


def test_repr_round_trips_through_stamped_args():
    m = _model("repr", seed=1)
    a = SimpleNamespace(**vars(parse_args(["--trunk", "qwen", "--mixer-size", "small", "--mixer-qk-norm", "--mixer-where", "repr"])))
    rebuilt = EZV2Model(4, obs_channels=12, trunk=a.trunk, mixer_size=a.mixer_size, mixer_qk_norm=a.mixer_qk_norm,
                        mixer_where=getattr(a, "mixer_where", "both")).eval()
    rebuilt.load_state_dict(m.state_dict())                       # strict: a missing or extra dyn_mixer key fails here
    obs = torch.rand(2, 12, 96, 96)
    with torch.no_grad():
        assert torch.equal(rebuilt.representation(obs), m.representation(obs))
    with pytest.raises(RuntimeError):
        _model("both").load_state_dict(m.state_dict())            # a D1 checkpoint does not load into the full model


def test_repr_needs_a_mixer_trunk():
    m = EZV2Model(4, obs_channels=12, trunk="resnet", mixer_where="repr")   # the model ignores it without a mixer; the trainer asserts
    assert not hasattr(m, "dyn_mixer")
    with pytest.raises(AssertionError):
        EZV2Model(4, obs_channels=12, trunk="qwen", mixer_where="dynamics")
