"""TODO A71 (2026-11-01): the Qwen Q-critic head - opt-in, trained on the counterfactual Q labels, and actually read by act().

The Gemini review of 2026-11-01 found that `rail_head` is trained but never read at evaluation, so a Q head that act() ignores cannot change a
single closed-loop route. These tests pin the opposite: the head exists only when asked for, its labels and loss mask frames without a label,
the plain decode is untouched unless WOR_QCRITIC is set, and with it set the Q values move both the commanded speed and the lateral offset.
"""
import dataclasses
import gzip
import json
import os

import numpy as np
import pytest
import torch

from src.models.world_on_rails.q_critic import (
    LAT_TARGETS, QCriticHead, decode_with_q, parse_qcritic_env, q_critic_loss, q_critic_term)
from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
from src.models.world_on_rails.aux_heads import TARGET_SPEEDS
from src.training.counterfactual_q import QConfig

KW = dict(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True, route_points=4)
NV, ND = len(TARGET_SPEEDS), len(LAT_TARGETS)


def _inputs(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(rgb=torch.rand(b, 3, 96, 256, generator=g), speed=torch.rand(b, 1, generator=g) * 10, command=torch.zeros(b, dtype=torch.long),
                route=torch.randn(b, 4, 2, generator=g))


# ---------------------------------------------------------------- head and loss

def test_q_head_is_opt_in_zero_initialised_and_shaped_speed_by_lateral():
    assert "q_sa" not in QwenWorldOnRailsPolicy(**KW).eval()(**_inputs())
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    out = m(**_inputs())
    assert out["q_sa"].shape == (3, NV, ND)
    assert torch.count_nonzero(out["q_sa"]) == 0, "zero-initialised: a resumed E15 starts with Q = 0 everywhere"


def test_loss_masks_frames_without_labels_and_uses_huber():
    pred = torch.zeros(3, NV, ND, requires_grad=True)
    target = torch.full((3, NV, ND), 2.0)
    target[1] = float("nan")                                     # a frame with no label (tail of a route, or a recovery-camera frame)
    loss = q_critic_loss(pred, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad[1].abs().sum() == 0 and pred.grad[0].abs().sum() > 0
    assert abs(float(loss) - 1.5) < 1e-5                         # smooth L1 with beta 1 at |error| = 2: 2 - 0.5
    assert float(q_critic_loss(pred, torch.full((3, NV, ND), float("nan")))) == 0.0


def test_trainer_term_reads_the_batch_and_respects_the_weight():
    out = {"q_sa": torch.zeros(2, NV, ND)}
    batch = {"target_qsa": torch.ones(2, NV * ND), "qsa_valid": torch.tensor([1.0, 0.0])}
    term = q_critic_term(out, batch, weight=2.0, device="cpu")
    assert term is not None and abs(float(term) - 2.0 * 0.5) < 1e-6        # only the valid row counts; smooth L1 at error 1 = 0.5
    assert q_critic_term(out, batch, weight=0.0, device="cpu") is None
    assert q_critic_term({}, batch, weight=1.0, device="cpu") is None


# ---------------------------------------------------------------- decoding

def _probs(v):
    p = np.full(NV, 1e-3)
    p[int(np.argmin(np.abs(np.array(TARGET_SPEEDS) - v)))] = 1.0
    return p / p.sum()


def test_decode_with_beta_zero_is_the_plain_mean_and_stays_in_lane():
    probs = _probs(10.0)
    q = np.random.default_rng(0).normal(size=(NV, ND))
    v, d, _ = decode_with_q(probs, q, np.array(TARGET_SPEEDS), beta=0.0)
    assert abs(v - float((probs * np.array(TARGET_SPEEDS)).sum())) < 1e-9 and d == 0.0


def test_decode_moves_speed_and_lateral_offset_toward_high_q():
    bins = np.array(TARGET_SPEEDS)
    probs = _probs(10.0)
    q = np.zeros((NV, ND))
    q[0, :] = 5.0                                                # stopping is worth far more (a stopped car ahead)
    v_slow, d, _ = decode_with_q(probs, q, bins, beta=5.0)
    assert v_slow < 1.0 and d == 0.0
    q = np.zeros((NV, ND))
    q[:, LAT_TARGETS.index(3.5)] = 5.0                           # the right lane is worth far more (a free lane past an obstacle)
    _, d, dbg = decode_with_q(probs, q, bins, beta=5.0)
    assert d == 3.5 and dbg["p_lat"][LAT_TARGETS.index(3.5)] > 0.5
    q = np.zeros((NV, ND))
    q[:, LAT_TARGETS.index(3.5)] = 0.05                          # a marginal preference must not pull the car out of its lane
    _, d, _ = decode_with_q(probs, q, bins, beta=5.0)
    assert d == 0.0


def test_env_parsing():
    assert parse_qcritic_env(None) is None and parse_qcritic_env("") is None
    assert parse_qcritic_env("0") is None and parse_qcritic_env("off") is None
    cfg = parse_qcritic_env("beta=2,lat_margin=1.5,lateral=0,hold_ticks=10")
    assert cfg["beta"] == 2.0 and cfg["lat_margin"] == 1.5 and cfg["lateral"] == 0.0 and cfg["hold_ticks"] == 10
    assert parse_qcritic_env("beta=1")["lateral"] == 1.0         # defaults fill in
    with pytest.raises(ValueError):
        parse_qcritic_env("beta=1,lat_thresh=0.9")               # a typo must not silently fall back to the default


def test_speed_is_conditioned_on_the_executed_lateral_offset():
    """Review 2026-11-01: an obstacle in the own lane, both side lanes free. The lateral mass splits between left and right, so the car stays
    in lane - and must then command the in-lane speed (stop), not the average over lane changes it does not make."""
    bins = np.array(TARGET_SPEEDS)
    probs = np.full(NV, 1.0 / NV)
    q = np.full((NV, ND), 5.0)
    q[1:, LAT_TARGETS.index(0.0)] = -10.0
    q[:, [0, 4]] = -10.0                                          # the far offsets are off-road
    v, d, _ = decode_with_q(probs, q, bins, beta=1.0)
    assert d == 0.0 and v < 0.5
    v, d, _ = decode_with_q(probs, q, bins, beta=1.0, lateral=False)
    assert d == 0.0 and v < 0.5
    v, d, _ = decode_with_q(probs, q, bins, beta=1.0, forced_d=-1.75)
    assert d == -1.75 and v > 5.0                                 # in the free left lane the car keeps driving


def test_lateral_shift_is_anchored_to_the_route_and_ramped():
    from src.models.world_on_rails.q_critic import shift_toward_lateral
    route = np.array([[2.5, 0.0], [7.5, 0.0], [12.5, 0.0], [17.5, 0.0]])
    bc = np.array([[2.0, -3.5], [4.0, -3.5], [6.0, -3.5], [8.0, -3.5], [10.0, -3.5]], np.float32)   # BC is already swerving left
    out = shift_toward_lateral(bc, route, d_target=-3.5, lat_rate=1.75)
    assert np.allclose(out, bc, atol=1e-4), "an offset is relative to the route, not added on top of the BC swerve (-7 m)"
    straight = np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]], np.float32)
    out = shift_toward_lateral(straight, route, d_target=3.5, lat_rate=1.75)
    steps = np.abs(np.diff(np.r_[0.0, out[:, 1]]))
    assert np.all(steps <= 1.75 * 0.25 + 1e-4) and out[-1, 1] > 1.5                # ramps at the label's lateral rate
    assert np.allclose(out[:, 0], straight[:, 0], atol=1e-4)                        # a residual shift along the normal: BC's longitudinal plan kept
    curved_bc = np.array([[2.0, 0.1], [4.0, 0.3], [6.0, 0.6], [8.0, 1.0], [10.0, 1.5]], np.float32)   # BC's own lane curvature
    out = shift_toward_lateral(curved_bc, route, d_target=-1.75, lat_rate=100.0)
    assert np.allclose(out[:, 1], -1.75, atol=1e-4)                                 # residual brings every point to the target offset
    standing = np.array([[0.05, 0.0], [0.1, 0.0], [0.15, 0.0], [0.2, 0.0], [0.25, 0.0]], np.float32)
    out = shift_toward_lateral(standing, np.array([[0.3, -0.3], [5.3, -0.3], [10.3, -0.3], [15.3, -0.3]]), d_target=-3.5, lat_rate=1.75)
    assert np.all(out[:, 0] > -0.1), "standing still off-centre must not push the aim point behind the car"
    bend = np.array([[2.5, 0.0], [6.0, -2.0], [7.0, -6.0], [5.0, -10.0]])           # a hairpin to the left (route x not monotone)
    wps = np.array([[2.0, 0.0], [4.0, -0.8], [5.5, -2.0], [6.5, -4.0], [6.8, -6.0]], np.float32)
    out = shift_toward_lateral(wps, bend, d_target=-1.75, lat_rate=7.0)
    from src.training.counterfactual_q import RouteFrame
    rf = RouteFrame(bend, from_origin=False)
    _, d_before = rf.project(wps[:, 0], wps[:, 1])
    _, d_after = rf.project(out[:, 0], out[:, 1])
    assert np.all(d_after - d_before < 0.0), "every waypoint moves left of the route, none to the right"


def test_lateral_shift_runs_without_the_training_stack():
    """2026-11-02: A71-on crashed routes in the CARLA venv, which has no tensorboard. The first confirmed lane change imported RouteFrame
    through src/training/__init__.py, which loaded the trainer, then src.logging, then tensorboard. The agent's import closure must stay
    numpy-only: with tensorboard unimportable the shift still runs and the trainer is never loaded."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['tensorboard'] = None\n"
            "import numpy as np\n"
            "from src.models.world_on_rails.q_critic import shift_toward_lateral\n"
            "out = shift_toward_lateral(np.array([[2.0, 0.0], [4.0, 0.0]], np.float32), np.array([[2.5, 0.0], [7.5, 0.0], [12.5, 0.0]]),"
            " d_target=1.75, lat_rate=1.75)\n"
            "assert out.shape == (2, 2) and out[1, 1] > 0.0\n"
            "assert 'src.training.wor_trainer' not in sys.modules and 'src.logging.experiment_logger' not in sys.modules\n"
            "print('SHIFT_OK')\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": root})
    assert r.returncode == 0 and "SHIFT_OK" in r.stdout, r.stderr[-1500:]


def test_training_package_still_exports_its_classes():
    """The lazy package init keeps `from src.training import WorldOnRailsTrainer` (and the dataset helpers) working."""
    from src.training import WorldOnRailsDataset, WorldOnRailsTrainer, create_wor_dataloader
    from src.training.wor_dataset import WorldOnRailsDataset as D
    from src.training.wor_trainer import WorldOnRailsTrainer as T
    assert WorldOnRailsTrainer is T and WorldOnRailsDataset is D and callable(create_wor_dataloader)


def test_lateral_choice_is_held_for_hold_ticks(monkeypatch):
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    right = torch.zeros(NV, ND)
    right[:, LAT_TARGETS.index(3.5)] = 10.0
    centre = torch.zeros(NV, ND)
    centre[:, LAT_TARGETS.index(0.0)] = 10.0
    _act(m, monkeypatch, "beta=5,hold_ticks=3,confirm_ticks=1")
    lats = []
    for k in range(6):
        with torch.no_grad():
            m.q_head.net[-1].bias.copy_((right if k == 0 else centre).reshape(-1))
        _act(m, monkeypatch, None, keep_state=True)
        lats.append(m.last_debug["q_lat"])
    assert lats[:3] == [3.5, 3.5, 3.5] and lats[3:] == [0.0, 0.0, 0.0]


def test_switch_without_q_head_is_an_error(monkeypatch):
    m = QwenWorldOnRailsPolicy(**KW).eval()
    with pytest.raises(RuntimeError):
        _act(m, monkeypatch, "beta=1")


def test_beta_zero_passes_the_plain_speed_through_exactly(monkeypatch):
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    assert _act(m, monkeypatch)["tgt"] == _act(m, monkeypatch, "beta=0")["tgt"]


# ---------------------------------------------------------------- act()

def _act(m, monkeypatch, env=None, keep_state=False):
    if not keep_state:
        if env is None:
            monkeypatch.delenv("WOR_QCRITIC", raising=False)
        else:
            monkeypatch.setenv("WOR_QCRITIC", env)
        for k in ("_qcritic_ready", "_safety_ready", "_q_lock"):
            m.__dict__.pop(k, None)
    seen = {}
    m.controller.control_from_waypoints = lambda waypoints, current_speed_kmh, target_speed_kmh=None: (
        seen.update(wps=np.array(waypoints), tgt=target_speed_kmh), (0.1, 0.5, 0.0))[1]
    rgb = np.random.default_rng(1).integers(0, 255, (96, 256, 3), dtype=np.uint8)
    m.act(rgb=rgb, speed=8.0, command=3, device="cpu", route=np.array([[2.5, 0.0], [7.5, 0.0], [12.5, 0.0], [17.5, 0.0]], np.float32))
    return seen


def test_act_is_unchanged_when_the_switch_is_off(monkeypatch):
    torch.manual_seed(0)
    plain = QwenWorldOnRailsPolicy(**KW).eval()
    torch.manual_seed(0)
    withq = QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    withq.load_state_dict(plain.state_dict(), strict=False)
    a, b = _act(plain, monkeypatch), _act(withq, monkeypatch)
    assert a["tgt"] == b["tgt"] and np.array_equal(a["wps"], b["wps"])


def test_act_reads_the_q_head_when_switched_on(monkeypatch):
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    off = _act(m, monkeypatch)
    with torch.no_grad():
        bias = torch.zeros(NV, ND)
        bias[0, :] = 10.0                                        # Q says: stop
        m.q_head.net[-1].bias.copy_(bias.reshape(-1))
    slow = _act(m, monkeypatch, "beta=5")
    assert slow["tgt"] < off["tgt"] and slow["tgt"] < 3.6
    with torch.no_grad():
        bias = torch.zeros(NV, ND)
        bias[:, LAT_TARGETS.index(3.5)] = 10.0                   # Q says: move one lane right
        m.q_head.net[-1].bias.copy_(bias.reshape(-1))
    right = _act(m, monkeypatch, "beta=5,confirm_ticks=1")
    ramp = np.minimum(3.5, off["wps"][:, 1] + 1.75 * 0.25 * (np.arange(5) + 1))      # straight route: the normal is +y; at most lat_rate * t_k
    assert np.allclose(right["wps"][:, 1], ramp, atol=1e-4)
    assert np.allclose(right["wps"][:, 0], off["wps"][:, 0], atol=1e-4)
    assert m.last_debug["q_lat"] == 3.5 and m.last_debug["q_sa"].shape == (NV, ND)
    no_lat = _act(m, monkeypatch, "beta=5,lateral=0,confirm_ticks=1")
    assert np.allclose(no_lat["wps"], off["wps"], atol=1e-6)


def test_per_tick_decode_log_is_written_only_when_asked(monkeypatch, tmp_path):
    """Route 2664 could not be diagnosed after the fact: nothing recorded what the decoder chose each tick."""
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    log = tmp_path / "decode.jsonl"
    monkeypatch.delenv("WOR_QCRITIC_LOG", raising=False)
    _act(m, monkeypatch, "beta=1")
    assert not log.exists()
    monkeypatch.setenv("WOR_QCRITIC_LOG", str(log))
    _act(m, monkeypatch, "beta=1")
    _act(m, monkeypatch, "beta=1", keep_state=True)
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(rows) == 2 and rows[1]["tick"] == rows[0]["tick"] + 1
    assert {"q_lat", "q_cand", "lock_left", "q_v", "v_lat", "q_sa"} <= set(rows[0]) and len(rows[0]["q_sa"]) == NV


# ---------------------------------------------------------------- resume and loader

def test_resuming_e15_style_checkpoint_into_a_q_critic_model_initialises_only_the_new_head():
    from src.training.wor_checkpoint import load_trainable_state
    src = QwenWorldOnRailsPolicy(**KW)
    dst = QwenWorldOnRailsPolicy(**KW, use_q_critic=True)
    frozen = [k for k in dst.state_dict() if k.startswith("encoder.")]
    state = {k: v for k, v in src.state_dict().items() if not k.startswith("encoder.")}
    changed = load_trainable_state(dst, state, frozen, "e15-like")
    assert changed and all(k.startswith("q_head.") for k in changed)


def test_loader_rebuilds_the_q_head_from_the_checkpoint_config(tmp_path, capsys):
    from src.models.world_on_rails.wor_loader import load_wor_model
    m = QwenWorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True,
                               route_points=4, use_q_critic=True)
    with torch.no_grad():
        m.q_head.net[-1].bias.normal_()
    cfg = {"backbone": "resnet34", "route_points": 4, "vision_grid": [4, 4], "use_target_speed": True, "target_speed_input": "policy",
           "use_rail_q": True, "use_ray_geometry": False, "crop_bottom_frac": 0.0, "use_q_critic": True}
    path = tmp_path / "ckpt.pth"
    torch.save({"epoch": 1, "model": m.state_dict(), "config": cfg}, path)
    loaded = load_wor_model(checkpoint_path=str(path), backbone_name="resnet34", pretrained_backbone=False, device="cpu", policy_arch="qwen10m")
    assert getattr(loaded, "q_head", None) is not None
    assert torch.allclose(loaded.q_head.net[-1].bias, m.q_head.net[-1].bias)


# ---------------------------------------------------------------- dataset labels

def _write_route(root, n=14):
    rd = root / "Town01" / "Route0_Rep0"
    (rd / "measurements").mkdir(parents=True)
    (rd / "rgb").mkdir()
    from PIL import Image
    for i in range(n):
        fid = f"{i:04d}"
        meas = {"speed": 5.0, "target_speed": 5.0, "command": 4, "ego_matrix": [[1, 0, 0, 1.25 * i], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "route": [[2.5 + j, 0.0] for j in range(20)], "route_original": [[2.5 + j, 0.0] for j in range(20)]}
        with gzip.open(rd / "measurements" / f"{fid}.json.gz", "wt") as f:
            json.dump(meas, f)
        Image.fromarray(np.zeros((32, 64, 3), np.uint8)).save(rd / "rgb" / f"{fid}.jpg")
    q = np.arange(n * NV * ND, dtype=np.float32).reshape(n, NV, ND)
    q[n - 3:] = np.nan
    np.savez(rd / "qlab_test.npz", frame=np.array([f"{i:04d}" for i in range(n)]), q=q, config=json.dumps(dataclasses.asdict(QConfig())))
    return rd, q


def test_dataset_attaches_q_labels_by_frame_id(tmp_path, monkeypatch):
    from src.training.wor_dataset import WorldOnRailsDataset
    _, q = _write_route(tmp_path)
    monkeypatch.setenv("WOR_Q_LABELS", "qlab_test")
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)
    assert len(ds) > 0
    for i in range(len(ds)):
        s = ds[i]
        fid = int(os.path.basename(ds.samples[i]["rgb_path"]).split(".")[0])
        assert s["target_qsa"].shape == (NV * ND,)
        if np.isfinite(q[fid]).all():
            assert float(s["qsa_valid"]) == 1.0 and np.allclose(s["target_qsa"].numpy(), q[fid].reshape(-1))
        else:
            assert float(s["qsa_valid"]) == 0.0


def test_validation_reports_q_loss_as_a_diagnostic_only(tmp_path, monkeypatch):
    """val_q_critic_loss on held-out frames, weighted by labelled rows; val_loss (best-model selection) stays the waypoint objective."""
    from torch.utils.data import DataLoader
    from src.training.wor_dataset import WorldOnRailsDataset
    from src.training.wor_eval import run_validation
    from contextlib import nullcontext
    _write_route(tmp_path)
    monkeypatch.setenv("WOR_Q_LABELS", "qlab_test")
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(96, 256), cache_decoded=False)
    loader = DataLoader(ds, batch_size=2)
    torch.manual_seed(0)
    plain = QwenWorldOnRailsPolicy(**KW)
    torch.manual_seed(0)
    withq = QwenWorldOnRailsPolicy(**KW, use_q_critic=True)
    withq.load_state_dict(plain.state_dict(), strict=False)
    a = run_validation(plain, loader, "cpu", lambda enabled: nullcontext(), False)
    b = run_validation(withq, loader, "cpu", lambda enabled: nullcontext(), False)
    assert "val_q_critic_loss" not in a and b["val_q_critic_loss"] > 0
    assert abs(a["val_loss"] - b["val_loss"]) < 1e-6


def test_dataset_refuses_to_train_on_missing_labels(tmp_path, monkeypatch):
    """A wrong tag or data path used to give zero labelled frames, a loss of exactly 0.0 and a Q head that stays at zero - a silent no-op run."""
    from src.training.wor_dataset import WorldOnRailsDataset
    _write_route(tmp_path)
    monkeypatch.setenv("WOR_Q_LABELS", "no_such_tag")
    with pytest.raises(RuntimeError, match="Q labels"):
        WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)


def test_label_file_with_another_action_grid_is_rejected(tmp_path):
    from src.training.counterfactual_q import load_q_labels
    rd = tmp_path / "r"
    rd.mkdir()
    np.savez(rd / "bad.npz", frame=np.array(["0000"]), q=np.zeros((1, 8, 3), np.float32))
    with pytest.raises(ValueError):
        load_q_labels(str(rd), "bad")


def test_dataset_without_the_switch_emits_invalid_labels(tmp_path, monkeypatch):
    from src.training.wor_dataset import WorldOnRailsDataset
    _write_route(tmp_path)
    monkeypatch.delenv("WOR_Q_LABELS", raising=False)
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)
    s = ds[0]
    assert float(s["qsa_valid"]) == 0.0 and s["target_qsa"].shape == (NV * ND,)


def _labels_with_config(rd, tag, cfg_dict):
    rd.mkdir(exist_ok=True)
    np.savez(rd / f"{tag}.npz", frame=np.array(["0000"]), q=np.zeros((1, NV, ND), np.float32), config=json.dumps(cfg_dict))


def test_label_file_built_with_another_hold_or_margin_is_rejected(tmp_path):
    """A stale tag (e.g. qcf_v2: hold 8, no overspeed_margin) used to load silently because only the action grid was compared."""
    import dataclasses
    from src.training.counterfactual_q import QConfig, load_q_labels
    good = dataclasses.asdict(QConfig())
    _labels_with_config(tmp_path / "ok", "t", good)
    assert len(load_q_labels(str(tmp_path / "ok"), "t")) == 1
    for key, val in (("hold", 8), ("margin", 0.3), ("gamma", 0.9), ("horizon", 8)):
        _labels_with_config(tmp_path / key, "t", {**good, key: val})
        with pytest.raises(ValueError, match=key):
            load_q_labels(str(tmp_path / key), "t")
    old = {k: v for k, v in good.items() if k != "overspeed_margin"}      # written before the field existed
    _labels_with_config(tmp_path / "old", "t", old)
    with pytest.raises(ValueError, match="overspeed_margin"):
        load_q_labels(str(tmp_path / "old"), "t")


def test_label_file_without_a_config_is_rejected(tmp_path):
    from src.training.counterfactual_q import load_q_labels
    rd = tmp_path / "r"
    rd.mkdir()
    np.savez(rd / "noc.npz", frame=np.array(["0000"]), q=np.zeros((1, NV, ND), np.float32))
    with pytest.raises(ValueError, match="config"):
        load_q_labels(str(rd), "noc")


def test_lane_change_needs_lat_margin_of_column_value():
    bins = np.array(TARGET_SPEEDS)
    q = np.zeros((NV, ND))
    q[:, LAT_TARGETS.index(-1.75)] = 0.9
    assert decode_with_q(_probs(10.0), q, bins, beta=1.0, lat_margin=1.0)[1] == 0.0
    q[3, LAT_TARGETS.index(-1.75)] = 1.1                          # one speed in the left lane is worth more than the margin
    assert decode_with_q(_probs(10.0), q, bins, beta=1.0, lat_margin=1.0)[1] == -1.75
    assert decode_with_q(_probs(10.0), q, bins, beta=1.0, lat_margin=1.5)[1] == 0.0


# ---------------------------------------------------------------- review 2026-11-01 evening: hysteresis on value, no sign flips, speed in a lane change

def test_shifted_car_stays_while_the_side_is_better_and_returns_without_a_margin():
    bins = np.array(TARGET_SPEEDS)
    q = np.zeros((NV, ND))
    q[:, LAT_TARGETS.index(-3.5)] = 2.0
    q[:, LAT_TARGETS.index(0.0)] = 1.5                            # still beside the obstacle: the side is better, but by less than the margin
    assert decode_with_q(_probs(8.0), q, bins, beta=1.0, lat_margin=1.0, current_d=-3.5)[1] == -3.5
    q[:, LAT_TARGETS.index(0.0)] = 2.1                            # past it: the own lane is best again
    assert decode_with_q(_probs(8.0), q, bins, beta=1.0, lat_margin=1.0, current_d=-3.5)[1] == 0.0


def test_no_direct_flip_from_one_side_to_the_other():
    bins = np.array(TARGET_SPEEDS)
    q = np.zeros((NV, ND))
    q[:, LAT_TARGETS.index(3.5)] = 5.0
    assert decode_with_q(_probs(8.0), q, bins, beta=1.0, current_d=-3.5)[1] == 0.0


def test_lane_change_speed_is_the_lowest_near_best_bin_not_a_softmax_over_ties():
    """Review: with bins >= 10 m/s tied (unreachable within the hold) softmax(Q) commanded 13 m/s against an expert target of 8, and an all-zero Q
    column 11 m/s."""
    bins = np.array(TARGET_SPEEDS)
    q = np.zeros((NV, ND))
    q[:, LAT_TARGETS.index(-3.5)] = [1.70, 2.34, 3.38, 3.23, 3.23, 3.23, 3.23, 3.23]
    q[:, LAT_TARGETS.index(0.0)] = -5.0
    v, d, _ = decode_with_q(_probs(0.0), q, bins, beta=1.0)
    assert d == -3.5 and v == pytest.approx(8.0)
    v, _, _ = decode_with_q(_probs(8.0), np.zeros((NV, ND)), bins, beta=1.0, forced_d=-3.5)
    assert v == pytest.approx(8.0)                               # a flat column: BC decides (it also stops for a red light while shifted)
    v, _, _ = decode_with_q(_probs(0.0), np.zeros((NV, ND)), bins, beta=1.0, forced_d=-3.5)
    assert v == 0.0


def test_a_one_tick_candidate_does_not_move_the_car(monkeypatch):
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_q_critic=True).eval()
    with torch.no_grad():
        bias = torch.zeros(NV, ND)
        bias[:, LAT_TARGETS.index(3.5)] = 10.0
        m.q_head.net[-1].bias.copy_(bias.reshape(-1))
    _act(m, monkeypatch, "beta=5,confirm_ticks=3")
    assert m.last_debug["q_lat"] == 0.0 and m.last_debug["q_cand"] == 3.5
    _act(m, monkeypatch, "beta=5,confirm_ticks=3", keep_state=True)
    assert m.last_debug["q_lat"] == 0.0
    _act(m, monkeypatch, "beta=5,confirm_ticks=3", keep_state=True)
    assert m.last_debug["q_lat"] == 3.5                          # the third agreeing tick commits


def test_a_shifted_car_returns_unless_its_offset_beats_the_lane_by_the_stay_margin():
    bins = np.array(TARGET_SPEEDS)
    q = np.zeros((NV, ND))
    q[:, LAT_TARGETS.index(-3.5)], q[:, LAT_TARGETS.index(-1.75)], q[:, LAT_TARGETS.index(0.0)] = 0.30, 0.35, 0.20
    assert decode_with_q(_probs(8.0), q, bins, beta=1.0, current_d=-3.5, stay_margin=0.25)[1] == 0.0       # review: it used to stay at -3.5
    q[:, LAT_TARGETS.index(-3.5)] = 1.0
    assert decode_with_q(_probs(8.0), q, bins, beta=1.0, current_d=-3.5, stay_margin=0.25)[1] == -3.5


def test_non_finite_q_falls_back_to_the_plain_decode():
    bins = np.array(TARGET_SPEEDS)
    q = np.zeros((NV, ND))
    q[2, 1] = np.nan
    probs = _probs(8.0)
    v, d, dbg = decode_with_q(probs, q, bins, beta=1.0)
    assert d == 0.0 and v == pytest.approx(float((probs * bins).sum())) and dbg.get("nonfinite")
