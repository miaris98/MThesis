"""TODO A72 (2026-11-01): discrete offline RL (IQL / CQL) on the logged PDM-Lite transitions, on the Qwen policy token.

The critic is the A71 Q head over the 8 speed bins x 5 lateral offsets plus a V head. Each logged frame k carries the expert's action (its
target-speed bin and the lateral offset of its route shift) and the reward of the expert's real step k -> k+1 (A71's reward terms); the next
state is the logged frame k+1. IQL: expectile regression of V on the target Q of the expert action, TD on Q(s, a_exp) -> r + gamma V(s').
CQL: logsumexp over the 40 actions minus Q(s, a_exp). IQL's extracted policy pi ~ pi_BC exp(beta A) equals pi_BC exp(beta Q) (V cancels over
actions), which is A71's decoder: evaluation reuses WOR_QCRITIC.
"""
import gzip
import json
import math

import numpy as np
import pytest
import torch

from src.training.offline_rl import (OfflineRLConfig, cql_penalty, expectile_loss, offline_rl_terms, polyak_update)

NA = 40


# ---------------------------------------------------------------- losses

def test_expectile_weights_positive_and_negative_errors_by_tau():
    d = torch.tensor([2.0, -2.0])
    l = expectile_loss(d, 0.7)
    assert torch.allclose(l, torch.tensor([0.7 * 4.0, 0.3 * 4.0]))
    assert torch.allclose(expectile_loss(d, 0.5), torch.tensor([2.0, 2.0]))


def test_expectile_regression_converges_above_the_mean_for_tau_above_half():
    torch.manual_seed(0)
    x = torch.randn(4000)
    v = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([v], lr=0.05)
    for _ in range(600):
        opt.zero_grad()
        expectile_loss(x - v, 0.9).mean().backward()
        opt.step()
    assert float(v.detach()) > float(x.mean()) + 0.5                       # an upper expectile: IQL's in-sample max over actions


def test_cql_penalty_is_logsumexp_minus_the_data_action():
    q = torch.zeros(2, NA)
    q[0, 3] = 10.0
    a = torch.tensor([3, 3])
    pen = cql_penalty(q, a, torch.ones(2))
    assert pen.item() == pytest.approx(0.5 * (float(torch.logsumexp(q[0], 0) - 10.0) + math.log(NA)), rel=1e-5)


def _batch(B=4, r=1.0, done=0.0, valid=1.0):
    return {"exp_a": torch.arange(B) % NA, "exp_r": torch.full((B,), r), "has_next": torch.full((B,), 1.0 - done),
            "exp_valid": torch.full((B,), valid)}


def test_iql_td_target_uses_gamma_and_masks_terminal_and_unlabelled_rows():
    B = 4
    cfg = OfflineRLConfig(mode="iql", gamma=0.9, tau=0.7)
    q_sa = torch.zeros(B, NA, requires_grad=True)
    v = torch.zeros(B, requires_grad=True)
    v_next = torch.full((B,), 2.0)
    t = offline_rl_terms(q_sa, v, v_next, q_sa.detach(), _batch(B, r=1.0), cfg)
    assert t["rl_q"].item() == pytest.approx((1.0 + 0.9 * 2.0) - 0.5, rel=1e-5)     # Huber (beta 1) at |error| = 2.8
    t_done = offline_rl_terms(q_sa, v, v_next, q_sa.detach(), _batch(B, r=1.0, done=1.0), cfg)
    assert t_done["rl_q"].item() < t["rl_q"].item()              # no bootstrap past the last frame
    t_none = offline_rl_terms(q_sa, v, v_next, q_sa.detach(), _batch(B, valid=0.0), cfg)
    assert t_none["total"].item() == 0.0
    t["total"].backward()
    assert q_sa.grad is not None and v.grad is not None


def test_no_gradient_flows_into_the_bootstrap_or_the_target_q():
    B = 3
    cfg = OfflineRLConfig(mode="cql", cql_alpha=1.0)
    q_sa = torch.randn(B, NA, requires_grad=True)
    v = torch.randn(B, requires_grad=True)
    v_next = torch.randn(B, requires_grad=True)
    q_t = torch.randn(B, NA, requires_grad=True)
    offline_rl_terms(q_sa, v, v_next, q_t, _batch(B), cfg)["total"].backward()
    assert v_next.grad is None and q_t.grad is None
    assert "rl_cql" in offline_rl_terms(q_sa, v, v_next, q_t, _batch(B), cfg)


def test_polyak_update_moves_the_target_by_tau():
    src, tgt = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
        for p in tgt.parameters():
            p.fill_(0.0)
    polyak_update(tgt, src, 0.1)
    assert all(torch.allclose(p, torch.full_like(p, 0.1)) for p in tgt.parameters())


# ---------------------------------------------------------------- the expert's step reward (A71's reward terms)

def _scene_frames(n, speed=8.0, target_speed=8.0, y_after=0.0):
    """Straight driving along route_original (fixed in the world at y = 0); from frame 1 on the ego is `y_after` metres right of it."""
    out = []
    for t in range(n):
        x, y = speed * 0.25 * t, (y_after if t >= 1 else 0.0)
        meas = {"speed": speed, "target_speed": target_speed, "ego_matrix": [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0]],
                "route_original": [[2.5 + j, -y] for j in range(20)], "route": [[2.5 + j, -y] for j in range(20)]}
        boxes = [{"class": "ego_car", "extent": [2.45, 0.92, 0.75], "position": [0, 0, 0], "yaw": 0.0, "id": 1},
                 {"class": "ego_info", "num_lanes_same_direction": 1, "num_lanes_opposite_direction": 1, "ego_lane_number": 0, "is_in_junction": False}]
        out.append((meas, boxes))
    return out


def test_logged_step_reward_is_progress_minus_overspeed_and_lateral_cost():
    from src.training.counterfactual_q import QConfig, logged_step_reward
    cfg = QConfig()
    r = logged_step_reward(_scene_frames(cfg.horizon + 2), 0, cfg)
    assert r == pytest.approx(8.0 / cfg.v_max, abs=0.02)                         # 2 m in 0.25 s at w_progress 1 / (v_max dt)
    r_fast = logged_step_reward(_scene_frames(cfg.horizon + 2, speed=16.0, target_speed=8.0), 0, cfg)
    assert r_fast < 16.0 / cfg.v_max - 0.5 * cfg.w_overspeed * 7.5 / cfg.v_max      # over-speed costs
    r_off = logged_step_reward(_scene_frames(cfg.horizon + 2, y_after=4.5), 0, cfg)  # steps 4.5 m right of the route: past the shoulder
    assert r_off < r - 0.9 * cfg.w_offroad


# ---------------------------------------------------------------- label files carry the expert transition of every frame

def test_label_files_store_expert_action_and_reward_for_every_frame_even_when_strided(tmp_path):
    import importlib.util
    from pathlib import Path
    from src.training.counterfactual_q import QConfig, load_q_transitions
    spec = importlib.util.spec_from_file_location("bql", Path(__file__).resolve().parents[1] / "scripts" / "training" / "build_q_labels.py")
    bql = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bql)
    cfg = QConfig()
    rd = tmp_path / "Town01" / "data" / "X" / "Route0_Rep0"
    (rd / "measurements").mkdir(parents=True)
    (rd / "boxes").mkdir()
    for i, (m, b) in enumerate(_scene_frames(cfg.horizon + 8)):
        for sub, obj in (("measurements", m), ("boxes", b)):
            with gzip.open(rd / sub / f"{i:04d}.json.gz", "wt") as f:
                json.dump(obj, f)
    bql.label_route_dir(str(rd), "qt", cfg, stride=4)
    tr = load_q_transitions(str(rd), "qt")
    assert len(tr) == cfg.horizon + 7                                           # every frame with a logged successor, not only every 4th
    a, r = tr["0000"]
    v_idx, d_idx = divmod(a, len(cfg.lat_targets))
    assert cfg.speed_targets[v_idx] == 8.0 and cfg.lat_targets[d_idx] == 0.0 and r == pytest.approx(0.4, abs=0.02)


# ---------------------------------------------------------------- dataset transitions

def _write_pdm_route(root, n=14, tag="qtr"):
    from PIL import Image
    from src.training.counterfactual_q import QConfig
    import dataclasses
    rd = root / "Town01" / "Route0_Rep0"
    (rd / "measurements").mkdir(parents=True)
    (rd / "rgb").mkdir()
    for i in range(n):
        meas = {"speed": 5.0, "target_speed": 5.0, "command": 4, "ego_matrix": [[1, 0, 0, 1.25 * i], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "route": [[2.5 + j, 0.0] for j in range(20)], "route_original": [[2.5 + j, 0.0] for j in range(20)]}
        with gzip.open(rd / "measurements" / f"{i:04d}.json.gz", "wt") as f:
            json.dump(meas, f)
        Image.fromarray(np.full((32, 64, 3), i, np.uint8)).save(rd / "rgb" / f"{i:04d}.jpg")
    q = np.zeros((n, 8, 5), np.float32)
    q[1::2] = np.nan                                                            # strided Q labels
    np.savez(rd / f"{tag}.npz", frame=np.array([f"{i:04d}" for i in range(n)]), q=q, config=json.dumps(dataclasses.asdict(QConfig())),
             a_expert=np.arange(n) % NA, r_expert=(np.arange(n) / 13.0).astype(np.float32), stride=np.array(2))
    return rd


def test_dataset_returns_the_next_frame_and_the_expert_transition(tmp_path, monkeypatch):
    from src.training.wor_dataset import WorldOnRailsDataset
    _write_pdm_route(tmp_path)
    monkeypatch.setenv("WOR_Q_LABELS", "qtr")
    monkeypatch.setenv("WOR_Q_MIN_COVERAGE", "0.2")
    monkeypatch.setenv("WOR_TRANSITIONS", "1")
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)
    s0 = ds[0]
    frame0 = int(ds.samples[0]["rgb_path"][-8:-4])
    assert float(s0["has_next"]) == 1.0 and float(s0["exp_valid"]) == 1.0
    assert int(s0["exp_a"]) == frame0 % NA and float(s0["exp_r"]) == pytest.approx(frame0 / 13.0, abs=1e-5)
    assert s0["next_rgb"].shape == s0["rgb"].shape and s0["next_rgb"].float().mean() > s0["rgb"].float().mean()   # frame k+1 is brighter
    last = ds[len(ds) - 1]
    assert float(last["has_next"]) == 0.0 and last["next_rgb"].shape == last["rgb"].shape
    monkeypatch.delenv("WOR_TRANSITIONS")
    assert "next_rgb" not in WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)[0]


# ---------------------------------------------------------------- model and trainer

KW = dict(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True, route_points=4)


def test_v_head_is_opt_in_zero_initialised_and_resumable():
    from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
    from src.training.wor_checkpoint import load_trainable_state
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True, use_v_head=True).eval()
    out = m(torch.zeros(2, 96, 256, 3, dtype=torch.uint8), torch.zeros(2, 1), torch.zeros(2, dtype=torch.long), torch.zeros(2, 4, 2))
    assert out["v_s"].shape == (2,) and torch.all(out["v_s"] == 0) and out["policy_repr"].shape[0] == 2
    assert "v_s" not in QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()(
        torch.zeros(1, 96, 256, 3, dtype=torch.uint8), torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), torch.zeros(1, 4, 2))
    src = QwenWorldOnRailsPolicy(**KW, use_q_critic=True)
    frozen = [k for k in m.state_dict() if k.startswith("encoder.")]
    changed = load_trainable_state(m, {k: v for k, v in src.state_dict().items() if not k.startswith("encoder.")}, frozen, "a71-like")
    assert changed and all(k.startswith("v_head.") for k in changed)


def test_one_iql_training_step_runs_and_moves_the_target_q(tmp_path, monkeypatch):
    from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
    from src.training.wor_trainer import WorldOnRailsTrainer
    _write_pdm_route(tmp_path, n=40)                                            # 14 frames index only ~2 samples: no full batch
    monkeypatch.setenv("WOR_Q_LABELS", "qtr")
    monkeypatch.setenv("WOR_Q_MIN_COVERAGE", "0.2")
    monkeypatch.setenv("WOR_TRANSITIONS", "1")
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True, use_v_head=True)
    tr = WorldOnRailsTrainer(m, data_dir=str(tmp_path), save_dir=str(tmp_path / "ck"), batch_size=4, num_workers=0, device="cpu",
                             img_size=(96, 256), use_mlflow=False, val_split=0.0, q_critic_loss_weight=1.0, target_speed_loss_weight=0.2,
                             offline_rl=OfflineRLConfig(mode="iql"))
    before = [p.detach().clone() for p in tr.q_target.parameters()]
    metrics = tr.train_epoch(1)
    assert metrics["rl_q_loss"] > 0.0
    assert math.isfinite(metrics["rl_q_loss"]) and math.isfinite(metrics["rl_v_loss"])
    with torch.no_grad():                                                       # zero-initialised q_head moved, so the EMA target did too
        assert any(not torch.equal(a, b) for a, b in zip(before, tr.q_target.parameters()))


def test_train_wor_refuses_offline_rl_without_the_q_head_and_labels():
    import importlib.util
    import types
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("train_wor", Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_wor.py")
    tw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tw)
    with pytest.raises(SystemExit):
        tw.check_offline_rl_args(types.SimpleNamespace(offline_rl="iql", q_critic=0, q_labels=None))
    with pytest.raises(SystemExit):
        tw.check_offline_rl_args(types.SimpleNamespace(offline_rl="cql", q_critic=1, q_labels=None))
    tw.check_offline_rl_args(types.SimpleNamespace(offline_rl="iql", q_critic=1, q_labels="qcf_v4"))
    tw.check_offline_rl_args(types.SimpleNamespace(offline_rl="none", q_critic=0, q_labels=None))
