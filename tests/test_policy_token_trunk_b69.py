"""TODO B69: --mixer-where pred, the CARLA trunk's layout as the Atari prediction function.

CARLA (QwenWaypointTransformer): CNN features -> one token per grid cell (projection + learned position) -> a learned policy token in
front -> Qwen blocks -> final RMSNorm -> Linear-GELU-Linear heads on the policy token. Under "pred" the EZ-V2 conv representation and
dynamics play the CNN encoder's role and the trunk reads the value and policy from its policy token. The 30m trunk's blocks must be
CARLA's qwen30m blocks exactly (the same-network rule), the policy token must stay out of weight decay as in CARLA's wor_optim, the
heads must start at zero as EZ-V2's do, and the model must round-trip through its stamped args."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from atari_qwen.models.ez_model import EZV2Model, PolicyTokenTrunk
from atari_qwen.training.train_ez_offpolicy import mixer_param_groups, parse_args
from src.models.world_on_rails.qwen_wor_policy import QwenWaypointTransformer, _MODEL_SIZES
from src.training.wor_optim import NO_DECAY_SUFFIXES


def _model(size="small", seed=0, **kw):
    torch.manual_seed(seed)
    return EZV2Model(4, obs_channels=12, trunk="qwen", mixer_size=size, mixer_qk_norm=True, mixer_where="pred", **kw).eval()


def _pred_stats():
    """mixer_diag switches autograd off for the whole process when imported (it is a CPU diagnostic script); restore it."""
    grad = torch.is_grad_enabled()
    from atari_qwen.scripts.mixer_diag import pred_stats
    torch.set_grad_enabled(grad)
    return pred_stats


def _open_heads(m, std=0.02):
    for head in (m.pred_mixer.value_head, m.pred_mixer.policy_head):
        torch.nn.init.normal_(head[-1].weight, std=std)


def test_layout_replaces_the_conv_heads_and_the_mixers():
    m = _model()
    assert m.repr_mixer is None and m.dyn_mixer is None and isinstance(m.pred_mixer, PolicyTokenTrunk)
    keys = set(m.state_dict())
    assert not any(k.startswith(("pred_res.", "v_conv.", "p_conv.", "v_fc.", "p_fc.", "repr_mixer.", "dyn_mixer.")) for k in keys)
    assert {"pred_mixer.policy_token", "pred_mixer.pos"} <= keys
    assert m.pred_mixer.pos.shape == (1, 36, 128) and m.pred_mixer.policy_token.shape == (1, 1, 128)


def test_30m_blocks_are_the_carla_qwen30m_blocks():
    m = _model("30m")
    c = _MODEL_SIZES["30m"]
    carla = QwenWaypointTransformer(c["embed_dim"], c["depth"], c["num_heads"], c["ffn_dim"], qk_norm=True)
    shapes = lambda blocks: [(n, tuple(p.shape)) for n, p in blocks.named_parameters()]  # noqa: E731
    assert shapes(m.pred_mixer.blocks) == shapes(carla.blocks)
    assert type(m.pred_mixer.blocks[0]) is type(carla.blocks[0])
    assert type(m.pred_mixer.norm_out) is type(carla.final_norm)


def test_heads_start_at_zero_like_ezv2():
    m = _model()
    with torch.no_grad():
        s, v, p = m.initial_inference(torch.rand(3, 12, 96, 96))
    assert torch.equal(v, torch.zeros_like(v)) and torch.equal(p, torch.zeros_like(p))   # uniform prior, zero value


def test_representation_and_dynamics_are_ezv2():
    m = _model()
    obs, a = torch.rand(2, 12, 96, 96), torch.tensor([1, 3])
    with torch.no_grad():
        s = m.representation(obs)
        assert torch.equal(s, m.repr_res(m.down(obs)))
        plane = F.relu(m.act_ln(m.act_conv((a.float() / m.action_dim).view(-1, 1, 1, 1).expand(-1, 1, m.hw, m.hw))))
        assert torch.equal(m.dynamics(s, a), m.dyn_res(F.relu(m.dyn_bn(m.dyn_conv(torch.cat([s, plane], 1))) + s)))


def test_policy_token_reads_every_cell_and_trains():
    m = _model().train()
    _open_heads(m)
    s = torch.rand(2, 64, 6, 6, requires_grad=True)
    v, p = m.prediction(s)
    (v.square().mean() + p.square().mean()).backward()
    assert (s.grad.abs().flatten(2).amax(1) > 0).all()            # every one of the 36 cells reaches the readout
    t = m.pred_mixer
    for q in (t.policy_token, t.pos, t.inp.weight, t.blocks[0].attn.q_proj.weight, t.value_head[0].weight):
        assert q.grad is not None and float(q.grad.abs().max()) > 0


def test_param_groups_match_carla_no_decay():
    m = _model()
    assert "policy_token" in NO_DECAY_SUFFIXES and "vision_pos" in NO_DECAY_SUFFIXES
    names = {id(q): n for n, q in m.named_parameters()}
    decay, no_decay = mixer_param_groups(m, 1e-4)
    nd = {names[id(q)] for q in no_decay["params"]}
    assert {"pred_mixer.policy_token", "pred_mixer.pos"} <= nd
    assert {names[id(q)] for q in decay["params"] + no_decay["params"]} == {n for n in names.values() if n.startswith("pred_mixer.")}


def test_round_trip_through_stamped_args_and_diag():
    m = _model(seed=1)
    _open_heads(m)
    a = SimpleNamespace(**vars(parse_args(["--trunk", "qwen", "--mixer-size", "small", "--mixer-qk-norm", "--mixer-where", "pred"])))
    rebuilt = EZV2Model(4, obs_channels=12, trunk=a.trunk, mixer_size=a.mixer_size, mixer_qk_norm=a.mixer_qk_norm,
                        mixer_where=a.mixer_where).eval()
    rebuilt.load_state_dict(m.state_dict())
    obs = torch.rand(2, 12, 96, 96)
    with torch.no_grad():
        for x, y in zip(rebuilt.initial_inference(obs), m.initial_inference(obs)):
            assert torch.equal(x, y)
        stats = _pred_stats()(m.pred_mixer, m.representation(obs))
    assert len(stats) == 2 and all(0.0 <= b["entropy"] <= 1.0 for b in stats)
    with pytest.raises(RuntimeError):
        EZV2Model(4, obs_channels=12, trunk="qwen", mixer_size="small", mixer_qk_norm=True).load_state_dict(m.state_dict())


def test_norm_heads_and_guards():
    m = _model(norm="heads")                                       # the LayerNorm swap skips the absent conv heads
    assert m.pred_mixer is not None
    with pytest.raises(AssertionError):
        EZV2Model(4, obs_channels=12, trunk="gtrxl", mixer_where="pred")
