"""TODO A73 (2026-11-01): latent world model on the frozen pooled features, trained offline on the PDM-Lite logs (no simulator).

Dynamics, reward and value in the space of the frozen regnety 4x4 pooled map (the A59 pooled cache), an ensemble of 5 next-latent heads whose
spread is the conservatism penalty, a policy over A71's 40-action grid improved by a one-step lookahead in the model plus a BC term (the data
are expert). These tests run on synthetic features with known dynamics; the offline gate (5-step latent error, ensemble disagreement on
counterfactual actions, reward collapse) is checked before any CARLA run.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from src.models.world_on_rails.latent_world_model import LatentWorldModel, WorldModelConfig
from src.training.latent_world_model_train import (WMLossConfig, lookahead_q, imagined_return, unroll_losses, policy_improvement_loss,
                                                   update_target)
from scripts.analysis.a73_world_model_gate import disagreement_ratio, latent_error_vs_copy, reward_collapse

C, GH, GW, NA = 12, 2, 2, 40
SMALL = WorldModelConfig(feat_channels=C, grid=(GH, GW), embed_dim=32, repr_depth=1, dyn_depth=1, num_heads=4, ffn_dim=64, n_ensemble=5,
                         trunk="qwen")


def _toy_batch(B=32, K=5, seen_actions=(0, 1, 2, 3), seed=0):
    """z_{t+1} = z_t + shift[a] (per action), reward = 1 for action 1 else 0; only `seen_actions` appear in the data."""
    g = torch.Generator().manual_seed(seed)
    shift = torch.randn(NA, C, GH, GW, generator=torch.Generator().manual_seed(123)) * 0.5
    z = torch.randn(B, C, GH, GW, generator=g)
    acts = torch.tensor(seen_actions)[torch.randint(len(seen_actions), (B, K), generator=g)]
    feats = [z]
    for k in range(K):
        z = z + shift[acts[:, k]]
        feats.append(z)
    rewards = (acts == 1).float()
    return torch.stack(feats, 1), acts, rewards, torch.ones(B, K)


def test_shapes_for_every_trunk():
    for trunk in ("qwen", "mlp", "conv"):
        cfg = WorldModelConfig(**{**SMALL.__dict__, "trunk": trunk})
        m = LatentWorldModel(cfg)
        s = m.encode(torch.randn(3, C, GH, GW))
        assert s.shape == (3, GH * GW, 32)
        s_mean, s_all, r = m.step(s, torch.tensor([0, 5, 39]))
        assert s_mean.shape == s.shape and s_all.shape == (5, 3, GH * GW, 32) and r.shape == (3,)
        assert m.value(s).shape == (3,) and m.policy_logits(s).shape == (3, NA)
        assert s_all.std(0).mean() > 0                                          # the heads start different: the spread means something


def test_qwen_trunk_uses_the_project_block_at_30m_size():
    from src.models.transformer.layers import QwenTransformerBlock
    m = LatentWorldModel(WorldModelConfig(feat_channels=C, grid=(GH, GW), trunk="qwen", size="30m", n_ensemble=2))
    blocks = [x for x in m.modules() if isinstance(x, QwenTransformerBlock)]
    assert len(blocks) >= 8 and blocks[0].ffn is not None and m.cfg.embed_dim == 512


def test_training_on_known_dynamics_lowers_the_unrolled_latent_and_reward_losses():
    torch.manual_seed(0)
    m, tgt = LatentWorldModel(SMALL), LatentWorldModel(SMALL)
    tgt.load_state_dict(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    feats, acts, rew, valid = _toy_batch()
    first = None
    for it in range(300):
        loss = unroll_losses(m, tgt, feats, acts, rew, valid, WMLossConfig())
        if first is None:
            first = {k: float(v.detach()) for k, v in loss.items()}
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        update_target(tgt, m, 0.05)
    last = {k: float(v) for k, v in unroll_losses(m, tgt, feats, acts, rew, valid, WMLossConfig()).items()}
    assert last["latent"] < 0.5 * first["latent"] and last["reward"] < 0.5 * first["reward"]
    assert all(math.isfinite(v) for v in last.values())


def test_lookahead_prefers_the_rewarded_action_and_penalises_disagreement():
    torch.manual_seed(0)
    m, tgt = LatentWorldModel(SMALL), LatentWorldModel(SMALL)
    tgt.load_state_dict(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    feats, acts, rew, valid = _toy_batch(B=64, seen_actions=(0, 1, 2, 3))
    for _ in range(300):
        loss = unroll_losses(m, tgt, feats, acts, rew, valid, WMLossConfig())
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        update_target(tgt, m, 0.05)
    with torch.no_grad():
        s = m.encode(feats[:, 0])
        q0, _ = lookahead_q(m, s, gamma=0.0, penalty=0.0)                      # the toy's reward depends on the action only
        assert (q0[:, :4].argmax(1) == 1).float().mean() > 0.8                  # among the actions it has seen, the rewarded one
        q, u = lookahead_q(m, s, gamma=0.95, penalty=0.0)
        assert torch.isfinite(q).all()
        assert u[:, 4:].mean() > u[:, :4].mean()                                # the ensemble disagrees more on actions never in the data
        q_pen, _ = lookahead_q(m, s, gamma=0.95, penalty=5.0)
        assert (q_pen.argmax(1) < 4).float().mean() > (q.argmax(1) < 4).float().mean() - 1e-6
        ret = imagined_return(m, s, horizon=5, gamma=0.95, penalty=1.0)
        assert ret.shape == (64,) and torch.isfinite(ret).all()


def test_policy_improvement_mixes_lookahead_target_and_bc():
    torch.manual_seed(0)
    m = LatentWorldModel(SMALL)
    s = m.encode(torch.randn(8, C, GH, GW))
    a_exp = torch.zeros(8, dtype=torch.long)
    l = policy_improvement_loss(m, s, a_exp, WMLossConfig(bc_weight=0.2, imag_weight=1.0, penalty=1.0))
    assert set(l) >= {"policy_imag", "policy_bc", "total"} and torch.isfinite(l["total"])
    l["total"].backward()
    assert m.policy_head.weight.grad is not None


def test_gate_metrics():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert reward_collapse(pred, pred)["corr"] == pytest.approx(1.0) and reward_collapse(pred, pred)["std_ratio"] == pytest.approx(1.0)
    assert reward_collapse(np.zeros(4), pred)["collapsed"] is True
    assert disagreement_ratio(np.array([0.1, 0.1]), np.array([0.3, 0.5])) == pytest.approx(4.0)
    err = latent_error_vs_copy(np.array([0.5, 1.0]), np.array([1.0, 1.0]))
    assert err == pytest.approx([0.5, 1.0])


# ---------------------------------------------------------------- windows from the pooled cache and the label files

def _write_cache_and_labels(root, n_frames=12, n_routes=1):
    import dataclasses
    from src.training.counterfactual_q import QConfig
    from src.training.pooled_feature_cache import cache_files
    routes = [f"Town01/data/X/Route{r}_Rep0" for r in range(n_routes)]
    keys = [f"{rt}/rgb/{i:04d}.jpg" for rt in routes for i in range(n_frames)]
    prefix = str(root / "pooled" / "cache")
    os.makedirs(root / "pooled")
    jp, fp, dp = cache_files(prefix)
    mm = np.memmap(fp, dtype=np.float16, mode="w+", shape=(len(keys), C, GH, GW))
    mm[:] = np.tile(np.arange(n_frames, dtype=np.float16), n_routes)[:, None, None, None]
    mm.flush()
    np.memmap(dp, dtype=np.uint8, mode="w+", shape=(len(keys),))[:] = 1
    json.dump({"keys": keys, "n": len(keys), "channels": C, "grid": [GH, GW]}, open(jp, "w"))
    a = (np.arange(n_frames) % NA).astype(np.int16)
    a[7] = -1                                                                   # frame 7 has no expert transition
    for rt in routes:
        rd = root / "data" / rt
        rd.mkdir(parents=True)
        np.savez(rd / "qt.npz", frame=np.array([f"{i:04d}" for i in range(n_frames)]), q=np.zeros((n_frames, 8, 5), np.float32),
                 config=json.dumps(dataclasses.asdict(QConfig())), a_expert=a, r_expert=np.ones(n_frames, np.float32), stride=np.array(4))
    return str(root / "data"), prefix


def test_window_dataset_reads_consecutive_frames_with_their_transitions(tmp_path):
    from src.training.latent_world_model_train import LatentWindowDataset
    data_dir, prefix = _write_cache_and_labels(tmp_path)
    ds = LatentWindowDataset(data_dir, [prefix], "qt", K=3)
    # windows of 3 transitions need frames f..f+3 with transitions at f, f+1, f+2 (frame 7 has none, frame 11 has no successor)
    starts = sorted(int(w[0][-8:-4]) for w in ds.windows)
    assert starts == [0, 1, 2, 3, 4, 8]
    item = ds[ds.windows.index(next(w for w in ds.windows if w[0].endswith("0002.jpg")))]
    assert item["feats"].shape == (4, C, GH, GW) and float(item["feats"][0].mean()) == 2.0 and float(item["feats"][3].mean()) == 5.0
    assert item["actions"].tolist() == [2, 3, 4] and item["rewards"].shape == (3,)


def test_train_script_runs_an_epoch_and_writes_a_gate_ready_checkpoint(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("twm", Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_world_model.py")
    twm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twm)
    data_dir, prefix = _write_cache_and_labels(tmp_path, n_frames=30)
    out = tmp_path / "ck"
    twm.main(["--data_dir", data_dir, "--pooled_cache", prefix, "--q_labels", "qt", "--K", "3", "--trunk", "mlp", "--embed_dim", "32",
              "--repr_depth", "1", "--dyn_depth", "1", "--num_heads", "4", "--ffn_dim", "64", "--epochs", "1", "--batch_size", "4",
              "--num_workers", "0", "--val_frac", "0.0", "--save_dir", str(out), "--use_mlflow", "0", "--device", "cpu"])
    ck = torch.load(out / "wm_epoch_001.pth", map_location="cpu", weights_only=False)
    assert {"model", "target", "model_config", "run_config", "val_routes", "metrics"} <= set(ck)
    assert ck["run_config"]["K"] == 3 and math.isfinite(ck["metrics"]["train_total"])
    m = LatentWorldModel(WorldModelConfig(**ck["model_config"]))
    m.load_state_dict(ck["model"])


def test_gate_runs_on_a_trained_checkpoint_with_held_out_routes(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("twm2", Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_world_model.py")
    twm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twm)
    from scripts.analysis.a73_world_model_gate import main as gate_main
    data_dir, prefix = _write_cache_and_labels(tmp_path, n_frames=30, n_routes=4)
    out = tmp_path / "ck"
    twm.main(["--data_dir", data_dir, "--pooled_cache", prefix, "--q_labels", "qt", "--K", "3", "--trunk", "mlp", "--embed_dim", "32",
              "--repr_depth", "1", "--dyn_depth", "1", "--num_heads", "4", "--ffn_dim", "64", "--epochs", "1", "--batch_size", "4",
              "--num_workers", "0", "--val_frac", "0.25", "--save_dir", str(out), "--use_mlflow", "0", "--device", "cpu"])
    res = gate_main(["--ckpt", str(out / "wm_epoch_001.pth"), "--out", str(tmp_path / "gate.json"), "--num_workers", "0"])
    assert res["windows"] > 0 and len(res["latent_vs_copy"]) == 3 and "gate_pass" in res and (tmp_path / "gate.json").exists()


def test_conv_trunk_a73w_learns_and_its_ensemble_disagrees_on_unseen_actions():
    """A73-W: conv dynamics on the pooled 4x4 map (the WoR-family network) instead of transformer blocks."""
    from torch import nn
    torch.manual_seed(0)
    cfg = WorldModelConfig(**{**SMALL.__dict__, "trunk": "conv"})
    m, tgt = LatentWorldModel(cfg), LatentWorldModel(cfg)
    assert any(isinstance(x, nn.Conv2d) for x in m.modules())
    tgt.load_state_dict(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    feats, acts, rew, valid = _toy_batch(B=64)
    first = float(unroll_losses(m, tgt, feats, acts, rew, valid, WMLossConfig())["latent"].detach())
    for _ in range(300):
        loss = unroll_losses(m, tgt, feats, acts, rew, valid, WMLossConfig())
        opt.zero_grad()
        loss["total"].backward()
        opt.step()
        update_target(tgt, m, 0.05)
    with torch.no_grad():
        assert float(unroll_losses(m, tgt, feats, acts, rew, valid, WMLossConfig())["latent"]) < 0.5 * first
        _, u = lookahead_q(m, m.encode(feats[:, 0]), gamma=0.95, penalty=0.0)
        assert u[:, 4:].mean() > 2.0 * u[:, :4].mean()


def test_h_step_lookahead_matches_a_manual_rollout_and_h1_is_the_one_step_rule():
    """A73: policy improvement over imagined rollouts (h = 5 in the TODO): Q_h(s, a) = first step with a, then h - 1 greedy-policy steps, each
    reward penalised by the ensemble spread, then gamma^h V."""
    torch.manual_seed(0)
    m = LatentWorldModel(SMALL).eval()
    s = m.encode(torch.randn(3, C, GH, GW))
    with torch.no_grad():
        q1, _ = lookahead_q(m, s, gamma=0.9, penalty=0.5)
        q1b, _ = lookahead_q(m, s, gamma=0.9, penalty=0.5, horizon=1)
        assert torch.allclose(q1, q1b)
        q3, _ = lookahead_q(m, s, gamma=0.9, penalty=0.5, horizon=3)
        a = torch.tensor([7, 7, 7])
        st, ret, disc = s, torch.zeros(3), 1.0
        for k in range(3):
            st2, s_all, r = m.step(st, a)
            ret = ret + disc * (r - 0.5 * s_all.std(0).mean((1, 2)))
            disc *= 0.9
            st = st2
            a = m.policy_logits(st).argmax(-1)
        assert torch.allclose(q3[:, 7], ret + disc * m.value(st), atol=1e-5)


def test_train_script_takes_the_imagination_horizon():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("twm3", Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_world_model.py")
    twm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twm)
    a = twm.parse_args(["--data_dir", "d", "--pooled_cache", "p", "--q_labels", "t", "--save_dir", "s", "--imag_horizon", "5"])
    assert a.imag_horizon == 5


def test_policy_improvement_on_a_sub_batch_keeps_bc_on_every_state():
    """pi_batch (2026-11-02): the h-step lookahead over 40 actions is ~90% of a step (3.99 s at batch 256 on the box), so the imagination
    target runs on the first pi_batch states (the loader shuffles, so a random subset); BC still uses the whole batch."""
    torch.manual_seed(0)
    m = LatentWorldModel(SMALL)
    s = m.encode(torch.randn(8, C, GH, GW))
    a_exp = torch.arange(8) % 3
    full, sub = WMLossConfig(horizon=2), WMLossConfig(horizon=2, pi_batch=4)
    l_sub = policy_improvement_loss(m, s, a_exp, sub)
    assert torch.allclose(l_sub["policy_imag"], policy_improvement_loss(m, s[:4], a_exp[:4], full)["policy_imag"], atol=1e-6)
    assert torch.allclose(l_sub["policy_bc"], policy_improvement_loss(m, s, a_exp, full)["policy_bc"], atol=1e-6)
    assert torch.allclose(policy_improvement_loss(m, s, a_exp, WMLossConfig(horizon=2, pi_batch=64))["total"],
                          policy_improvement_loss(m, s, a_exp, full)["total"], atol=1e-6)      # pi_batch >= B: the whole batch


def test_train_script_takes_pi_batch_and_tf32():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("twm4", Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_world_model.py")
    twm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twm)
    base = ["--data_dir", "d", "--pooled_cache", "p", "--q_labels", "t", "--save_dir", "s"]
    a = twm.parse_args(base)
    assert a.pi_batch == 0 and a.tf32 == 0                                         # defaults: the whole batch, plain fp32
    a = twm.parse_args(base + ["--pi_batch", "64", "--tf32", "1"])
    assert a.pi_batch == 64 and a.tf32 == 1
