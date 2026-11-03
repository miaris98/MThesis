"""TODO (2026-11-03): checkpoint round trip. A saved checkpoint must reload into a freshly built network with identical weights, and the
gate measurement on it must give identical numbers. A save or key mismatch that the gate would read as a collapse is caught here."""
import torch

from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport
from scripts.analysis.attn_entropy_measure import gate_attention


def _build(seed):
    torch.manual_seed(seed)
    return EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk="conv_transformer", norm="batch", state_hw=6,
                     mixer_out_init_std=0.0, mixer_bg_init=0.0, simnorm=0, mixer_pos="rope", mixer_size="mini",
                     mixer_qk_norm=False, mixer_where="pred", mixer_no_pos=False)


def test_a_saved_checkpoint_reloads_with_identical_weights_and_gate_numbers(tmp_path):
    src = _build(0)
    torch.save({"model": src.state_dict(), "updates": 3, "env_steps": 5}, tmp_path / "ck.pt")
    dst = _build(99)                                   # a different init, so the load must do the work
    ck = torch.load(tmp_path / "ck.pt", map_location="cpu", weights_only=False)
    dst.load_state_dict(ck["model"], strict=True)
    a, b = src.state_dict(), dst.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    obs = torch.rand(8, 12, 96, 96, generator=torch.Generator().manual_seed(2))
    la, pa = gate_attention(src, obs)
    lb, pb = gate_attention(dst, obs)
    assert list(la) == list(lb) and float(pa.mean()) == float(pb.mean())


def test_a_checkpoint_saved_in_eval_mode_loads_back_in_train_mode_unchanged(tmp_path):
    m = _build(0).eval()
    torch.save({"model": m.state_dict()}, tmp_path / "ck.pt")
    n = _build(1)
    n.load_state_dict(torch.load(tmp_path / "ck.pt", map_location="cpu", weights_only=False)["model"])
    assert all(torch.equal(m.state_dict()[k], n.state_dict()[k]) for k in m.state_dict())
