"""TODO A74 (2026-11-02): closed-loop RL over a maneuver head on top of E15.

E15 is behaviour cloning; A71 / A72 / A73 learned values from the logs and none of them improved the closed loop (A71's decode: -0.3 DS on
obstacle routes, -18 on the others). A74 learns *when to deviate* from E15 from the closed-loop reward itself: 9 maneuvers (lateral
{left lane, stay, right lane} x speed {E15's own, creep, stop}) chosen by a small head on the frozen trunk's policy token and trained with PPO
anchored to a prior that is E15 ~99% of the time. These tests pin the contract: the default maneuver is E15 bit for bit, the head starts at
E15, a non-default choice is held long enough to complete a lane change, exploration rises when the car is stuck, the reward counts route
progress and the leaderboard's collision factors once per event, and the PPO step moves probability toward advantaged actions.
"""
import json
import os

import numpy as np
import pytest
import torch

from src.models.world_on_rails.maneuver_rl import (
    DEFAULT_ACTION, MANEUVERS, CREEP_KMH, ManeuverController, ManeuverHead, RewardTracker, RouteTracker, apply_maneuver, rl_maneuver)
from src.training.maneuver_ppo import PPOConfig, gae, load_episodes, ppo_update, prior_log_probs


def _wps():
    return np.array([[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]], np.float32)


def _route():
    return np.array([[2.5, 0.0], [7.5, 0.0], [12.5, 0.0], [17.5, 0.0]], np.float32)


def test_maneuver_table_has_the_default_e15_action():
    assert len(MANEUVERS) == 9 and MANEUVERS[DEFAULT_ACTION] == (0.0, "bc")
    assert {m[0] for m in MANEUVERS} == {-3.5, 0.0, 3.5} and {m[1] for m in MANEUVERS} == {"bc", "creep", "stop"}


def test_head_starts_at_e15_for_any_input():
    torch.manual_seed(0)
    head = ManeuverHead(64)
    head.reset_to_default(margin=7.0)
    logits, v = head(torch.randn(32, 64) * 10)
    p = logits.softmax(-1)
    assert p.shape == (32, 9) and v.shape == (32,)
    assert torch.all(p[:, DEFAULT_ACTION] > 0.99), "at step 0 the agent must drive as E15"
    assert torch.allclose(v, torch.zeros(32))


def test_default_maneuver_is_e15_bit_for_bit():
    wps, route = _wps(), _route()
    out_wps, out_kmh = apply_maneuver(DEFAULT_ACTION, wps, route, 23.4)
    assert out_wps is wps and out_kmh == 23.4


def test_speed_modes():
    stop = MANEUVERS.index((0.0, "stop"))
    creep = MANEUVERS.index((0.0, "creep"))
    assert apply_maneuver(stop, _wps(), _route(), 30.0)[1] == 0.0
    assert apply_maneuver(creep, _wps(), _route(), 30.0)[1] == CREEP_KMH
    assert apply_maneuver(creep, _wps(), _route(), 2.0)[1] == 2.0            # creep never speeds E15 up
    assert apply_maneuver(stop, _wps(), _route(), None)[1] == 0.0            # a policy without a speed head still stops


def test_lateral_maneuvers_move_the_waypoints_toward_the_chosen_lane():
    left, right = MANEUVERS.index((-3.5, "bc")), MANEUVERS.index((3.5, "bc"))
    wl, kl = apply_maneuver(left, _wps(), _route(), 20.0)
    wr, _ = apply_maneuver(right, _wps(), _route(), 20.0)
    assert np.all(wl[:, 1] < 0.0) and np.all(wr[:, 1] > 0.0) and kl == 20.0   # +y = right (q_shift_lateral's convention)
    assert np.all(np.abs(np.diff(np.r_[0.0, wl[:, 1]])) <= 1.75 * 0.25 + 1e-4)  # ramped at the label's lateral rate
    assert np.allclose(wl[:, 0], _wps()[:, 0], atol=1e-4)                        # E15's longitudinal plan is kept


def test_route_tracker_counts_forward_progress_near_the_route_only():
    rt = RouteTracker(np.stack([np.linspace(0, 100, 101), np.zeros(101)], 1), max_dist=10.0)
    assert rt.length == pytest.approx(100.0)
    p, d = rt.update(0.0, 0.0)
    assert p == pytest.approx(0.0) and d == pytest.approx(0.0)
    p, d = rt.update(10.0, 2.0)
    assert p == pytest.approx(10.0) and abs(d) == pytest.approx(2.0)
    assert rt.update(5.0, 0.0)[0] == 0.0                     # driving backwards earns nothing
    assert rt.update(25.0, 30.0)[0] == 0.0                    # far off the route earns nothing (the leaderboard's rule)
    assert rt.update(25.0, 0.0)[0] == pytest.approx(15.0)


def test_route_tracker_window_does_not_jump_to_a_later_leg():
    out = np.stack([np.linspace(0, 50, 51), np.zeros(51)], 1)
    back = np.stack([np.linspace(50, 0, 51), np.full(51, 6.0)], 1)             # a U-turn 6 m to the side, back past the start
    rt = RouteTracker(np.concatenate([out, back]), max_dist=10.0)
    p, _ = rt.update(2.0, 4.0)                                                 # near the start, closer to the return leg
    assert p < 5.0, "a point near the return leg must not credit the whole first leg"


def test_reward_counts_progress_and_each_collision_once():
    rw = RewardTracker(route_length=200.0, collision_penalty=10.0, offlane_cost=0.0)
    assert rw.step(progress_m=20.0, offset_m=0.0, t=1.0) == pytest.approx(10.0)            # 10% of the route
    rw.collision("vehicle", other_id=7, t=2.0)
    r = rw.step(progress_m=20.0, offset_m=0.0, t=2.05)
    assert r == pytest.approx(10.0 * 0.6 - 10.0)                                           # later progress x 0.6, penalty once
    rw.collision("vehicle", other_id=7, t=2.5)                                              # the same contact: not a new event
    assert rw.step(progress_m=0.0, offset_m=0.0, t=2.6) == pytest.approx(0.0) and rw.multiplier == pytest.approx(0.6)
    rw.collision("walker", other_id=9, t=3.0)
    assert rw.multiplier == pytest.approx(0.3) and rw.events == {"vehicle": 1, "walker": 1, "layout": 0}


def test_reward_charges_driving_off_the_lane():
    rw = RewardTracker(route_length=100.0, collision_penalty=10.0, offlane_cost=0.05, offlane_m=2.0)
    assert rw.step(progress_m=0.0, offset_m=1.0, t=0.0) == 0.0
    assert rw.step(progress_m=0.0, offset_m=-3.0, t=0.05) == pytest.approx(-0.05)


def _controller(**kw):
    torch.manual_seed(0)
    head = ManeuverHead(16)
    head.reset_to_default(margin=7.0)
    return ManeuverController(head, **kw)


def test_controller_holds_a_non_default_maneuver_long_enough_for_a_lane_change(tmp_path):
    ctl = _controller(period=5, commit=40, eps=1.0, eps_stuck=1.0, seed=0)              # eps=1: always explore a non-default maneuver
    feat = torch.zeros(1, 16)
    acts = [ctl.step(feat, speed_mps=5.0) for _ in range(80)]
    assert acts[0] != DEFAULT_ACTION and len(set(acts[:40])) == 1, "an exploratory maneuver is committed for 40 ticks"
    assert len(ctl.decisions) == 2


def test_controller_decides_every_period_on_the_default_and_logs_the_episode(tmp_path):
    ctl = _controller(period=5, commit=40, eps=0.0, eps_stuck=0.0, seed=0)
    feat = torch.zeros(1, 16)
    for _ in range(20):
        ctl.step(feat, speed_mps=5.0)
        ctl.add_reward(0.5)
    assert len(ctl.decisions) == 4 and all(d["action"] == DEFAULT_ACTION for d in ctl.decisions)
    info = ctl.finish(str(tmp_path / "ep.npz"), {"route": "RouteScenario_1"})
    z = np.load(tmp_path / "ep.npz")
    assert z["feat"].shape == (4, 16) and z["action"].tolist() == [DEFAULT_ACTION] * 4
    assert np.allclose(z["reward"], 2.5) and np.allclose(z["dt"], 1.0) and np.all(z["logp"] <= 0.0)
    assert json.loads(str(z["info"]))["route"] == "RouteScenario_1" and info["return"] == pytest.approx(10.0)


def test_controller_explores_more_when_the_car_is_stuck():
    feat = torch.zeros(1, 16)
    moving, stuck = _controller(eps=0.0, eps_stuck=0.5, stuck_s=1.0, seed=1), _controller(eps=0.0, eps_stuck=0.5, stuck_s=1.0, seed=1)
    a_moving = [moving.step(feat, speed_mps=5.0) for _ in range(400)]
    a_stuck = [stuck.step(feat, speed_mps=0.0) for _ in range(400)]
    assert all(a == DEFAULT_ACTION for a in a_moving)
    assert any(a != DEFAULT_ACTION for a in a_stuck), "standing still for > stuck_s raises exploration"
    # the logged behaviour log-prob is the mixture's, so PPO's ratio stays correct
    explored = [d for d in stuck.decisions if d["action"] != DEFAULT_ACTION]
    assert explored and all(d["logp"] > np.log(1e-3) for d in explored)


def test_greedy_mode_never_samples():
    ctl = _controller(mode="eval", eps=1.0, eps_stuck=1.0, seed=0)
    feat = torch.zeros(1, 16)
    assert all(ctl.step(feat, speed_mps=0.0) == DEFAULT_ACTION for _ in range(200))


def test_hook_is_a_no_op_without_a_controller():
    class P:
        pass
    wps = _wps()
    out_wps, out_kmh = rl_maneuver(P(), wps, _route(), 17.0, 3.0)
    assert out_wps is wps and out_kmh == 17.0


def test_gae_discounts_by_elapsed_decision_units():
    r = np.array([1.0, 0.0, 2.0]); v = np.array([0.5, 0.2, 0.1]); dt = np.array([1.0, 8.0, 1.0])
    g, lam = 0.9, 0.95
    adv, ret = gae(r, v, dt, g, lam)
    d2 = 2.0 - 0.1
    d1 = 0.0 + g ** 8 * 0.1 - 0.2
    d0 = 1.0 + g * 0.2 - 0.5
    a2 = d2; a1 = d1 + g ** 8 * lam * a2; a0 = d0 + g * lam * a1
    assert np.allclose(adv, [a0, a1, a2]) and np.allclose(ret, adv + v)


def test_prior_kl_is_zero_at_the_start_and_ppo_moves_toward_the_advantaged_action():
    torch.manual_seed(0)
    head = ManeuverHead(8)
    head.reset_to_default(margin=7.0)
    prior = prior_log_probs(margin=7.0)
    n = 512
    feats = torch.randn(n, 8)
    target = MANEUVERS.index((-3.5, "bc"))
    with torch.no_grad():
        logits, v = head(feats)
    actions = torch.full((n,), target, dtype=torch.long)
    old_logp = logits.log_softmax(-1)[:, target]
    adv = torch.ones(n)
    ret = torch.ones(n)
    cfg = PPOConfig(epochs=4, minibatch=128, lr=3e-3, kl_coef=0.0, ent_coef=0.0)
    stats0 = ppo_update(head, feats, actions, old_logp, adv, ret, prior, PPOConfig(epochs=1, minibatch=n, lr=0.0))
    assert stats0["kl_prior"] == pytest.approx(0.0, abs=1e-5)
    p0 = head(feats)[0].softmax(-1)[:, target].mean().item()
    ppo_update(head, feats, actions, old_logp, adv, ret, prior, cfg)
    p1 = head(feats)[0].softmax(-1)[:, target].mean().item()
    assert p1 > p0, "positive advantage on a maneuver must raise its probability"


def test_episodes_round_trip_through_the_rollout_directory(tmp_path):
    for i in range(3):
        ctl = _controller(eps=0.0, eps_stuck=0.0, seed=i)
        for _ in range(10):
            ctl.step(torch.zeros(1, 16), speed_mps=5.0)
            ctl.add_reward(1.0)
        ctl.finish(str(tmp_path / f"ep{i}.npz"), {"route": f"r{i}"})
    eps = load_episodes(str(tmp_path))
    assert len(eps) == 3 and all(e["feat"].shape == (2, 16) for e in eps) and sorted(e["info"]["route"] for e in eps) == ["r0", "r1", "r2"]


def test_episode_glue_turns_route_progress_and_collisions_into_rewards(tmp_path):
    from src.models.world_on_rails.maneuver_rl import RLEpisode
    ctl = _controller(eps=0.0, eps_stuck=0.0, seed=0)
    route_xy = np.stack([np.linspace(0, 100, 201), np.zeros(201)], 1)
    ep = RLEpisode(ctl, {"log": str(tmp_path / "roll"), "head": None}, route_xy, "RouteScenario_7_rep0")
    feat = torch.zeros(1, 16)
    for k in range(10):                                        # 1 m per tick for 10 ticks: 10% of the route
        ep.tick(float(k), 0.0, 0.05 * k)                       # the reward of the last tick's transition, credited before the decision
        ctl.step(feat, speed_mps=5.0)
    r_coll = ep.tick(10.0, 0.0, 0.5, collisions=[("vehicle.tesla.model3", 42)])
    assert r_coll == pytest.approx(100 * 1.0 / 100 * 0.6 - 10.0)
    info = ep.finish()
    assert info["rc"] == pytest.approx(10.0) and info["collisions"]["vehicle"] == 1 and info["route"] == "RouteScenario_7_rep0"
    files = os.listdir(tmp_path / "roll")
    assert len(files) == 1 and files[0].startswith("RouteScenario_7_rep0_") and files[0].endswith(".npz")
    assert info["return"] == pytest.approx(9.0 + r_coll)       # the first tick precedes any decision, so its progress is not credited


def test_learner_iteration_consumes_episodes_and_publishes_the_head(tmp_path):
    from src.models.world_on_rails.maneuver_rl import load_head
    from scripts.training.train_maneuver_ppo import learner_iteration, init_head
    roll, out = tmp_path / "roll", tmp_path / "out"
    os.makedirs(roll)
    head = init_head(str(out), embed_dim=16, margin=12.0, seed=0)
    assert os.path.exists(out / "head_latest.pth") and load_head(str(out / "head_latest.pth"), 16).meta["iter"] == 0
    for i in range(4):
        ctl = ManeuverController(load_head(str(out / "head_latest.pth"), 16), eps=1.0, eps_stuck=1.0, seed=i)   # all deviations
        for _ in range(80):
            ctl.step(torch.zeros(1, 16), speed_mps=5.0)
            ctl.add_reward(0.1)
        ctl.finish(str(roll / f"ep{i}.npz"), {"route": f"r{i}", "rc": 50.0})
    assert learner_iteration(head, str(roll), str(out), it=1, min_decisions=100, cfg=PPOConfig(epochs=1, minibatch=64)) is None   # 8 < 100
    stats = learner_iteration(head, str(roll), str(out), it=1, min_decisions=4, cfg=PPOConfig(epochs=1, minibatch=64))
    assert stats["episodes"] == 4 and stats["decisions"] == 8 and stats["deviation_rate"] == 1.0 and stats["rc_mean"] == 50.0
    assert load_head(str(out / "head_latest.pth"), 16).meta["iter"] == 1 and os.path.exists(out / "head_iter0001.pth")
    assert not [f for f in os.listdir(roll) if f.endswith(".npz")] and len(os.listdir(roll / "consumed" / "iter0001")) == 4


def test_learner_drops_episodes_from_a_stale_head(tmp_path):
    from src.models.world_on_rails.maneuver_rl import load_head
    from scripts.training.train_maneuver_ppo import learner_iteration, init_head
    roll, out = tmp_path / "roll", tmp_path / "out"
    os.makedirs(roll)
    head = init_head(str(out), embed_dim=16, margin=12.0, seed=0)
    ctl = ManeuverController(load_head(str(out / "head_latest.pth"), 16), eps=0.0, eps_stuck=0.0, seed=0)   # head_iter 0
    for _ in range(20):
        ctl.step(torch.zeros(1, 16), speed_mps=5.0)
    ctl.finish(str(roll / "old.npz"), {"route": "r"})
    stats = learner_iteration(head, str(roll), str(out), it=5, min_decisions=1, cfg=PPOConfig(epochs=1, minibatch=64), max_lag=2)
    assert stats is None and os.listdir(roll / "stale")                 # iter 0 is 5 updates behind: set aside, never trained on


def test_policy_act_applies_the_controller_and_is_unchanged_without_it():
    from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True,
                               route_points=4).eval()
    import copy
    rgb = np.random.RandomState(0).rand(96, 256, 3).astype(np.float32)
    route = _route()
    base = copy.deepcopy(m).act(rgb, 5.0, device="cpu", route=route)          # fresh copies: the PID controller keeps state between calls
    again = copy.deepcopy(m).act(rgb, 5.0, device="cpu", route=route)
    assert base == again
    m.act(rgb, 5.0, device="cpu", route=route)
    assert m._last_policy_repr.shape == (1, m.embed_dim)
    m = copy.deepcopy(m)

    class Stop:
        decisions = []

        def step(self, feat, speed_mps):
            assert feat.shape == (1, m.embed_dim)
            return MANEUVERS.index((0.0, "stop"))
    m._rl = Stop()
    steer, throttle, brake = m.act(rgb, 5.0, device="cpu", route=route)
    assert throttle == 0.0 and brake > 0.0, "a stop maneuver must brake whatever E15 wanted"
