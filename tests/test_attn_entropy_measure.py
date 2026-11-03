"""TODO (2026-11-03): the checkpoint attention measurement, now in the repo. It rebuilds a checkpoint's network from its saved args and reads
the conv trunk's normalised attention entropy on replay states. These tests build a tiny checkpoint and replay file on the PC, so the
measurement that decides Gate 2 is itself tested."""
import numpy as np
import torch

from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport
from scripts.analysis.attn_entropy_measure import measure


def _tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    args = {"trunk": "conv_transformer", "norm": "batch", "frame_size": 96, "mixer_size": "mini", "mixer_where": "pred",
            "mixer_qk_norm": False, "simnorm": 0, "mixer_out_init_std": 0.0, "mixer_bg_init": 0.0, "mixer_pos": "rope",
            "mixer_no_pos": False}
    m = EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk="conv_transformer", norm="batch", state_hw=6,
                  mixer_out_init_std=0.0, mixer_bg_init=0.0, simnorm=0, mixer_pos="rope", mixer_size="mini",
                  mixer_qk_norm=False, mixer_where="pred", mixer_no_pos=False)
    ck = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "args": args, "updates": 7, "env_steps": 9}, ck)
    rp = tmp_path / "replay.npz"
    frames = np.random.default_rng(0).integers(0, 256, size=(12, 2, 3, 96, 96), dtype=np.uint8)
    np.savez(rp, frames=frames)
    return str(ck), str(rp)


def test_measure_reads_each_layer_of_the_rebuilt_network(tmp_path):
    ck, rp = _tiny_checkpoint(tmp_path)
    upd, env, layers, policy = measure(ck, rp, n=16)
    assert (upd, env) == (7, 9)
    assert len(layers) == 4                       # mini: depth 4
    assert all(0.0 <= h <= 1.0 for h in layers) and 0.0 <= float(np.mean(policy)) <= 1.0


def test_measure_fails_loudly_on_a_checkpoint_that_does_not_match_its_args(tmp_path):
    ck, rp = _tiny_checkpoint(tmp_path)
    d = torch.load(ck, weights_only=False)
    d["args"]["mixer_size"] = "small"             # wrong size: the state dict cannot load into it
    torch.save(d, ck)
    import pytest
    with pytest.raises(RuntimeError):
        measure(ck, rp, n=4)
