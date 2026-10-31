"""TODO B70: the trainer's play diagnostics (life length, reward-free streaks, parked share, action shares, acting-policy entropy)."""
import numpy as np

from atari_qwen.training.train_ez_offpolicy import PARKED_STEPS, play_add, play_summary, play_window


def test_empty_window_has_no_steps():
    assert play_window(4)["steps"] == 0


def test_counts_over_two_steps():
    w = play_window(4)
    uni = np.full((2, 4), 0.25)
    play_add(w, [1, 2], np.array([3, PARKED_STEPS]), uni, [])
    play_add(w, [1, 3], np.array([0, PARKED_STEPS + 1]), uni, [120, 80])
    s = play_summary(w)
    m = s["metrics"]
    assert w["steps"] == 4
    assert np.allclose([m[f"play/act_share_{i}"] for i in range(4)], [0, 0.5, 0.25, 0.25])
    assert m["play/max_reward_free_streak"] == PARKED_STEPS + 1
    assert m["play/parked_share"] == 0.5                       # 2 of 4 env steps inside a >= PARKED_STEPS streak
    assert np.isclose(m["play/search_pi_entropy"], np.log(4), atol=1e-6)
    assert m["play/life_length"] == 100.0
    assert s["life len"] == "100 (n 2)"


def test_no_life_ended_leaves_life_length_out():
    w = play_window(3)
    play_add(w, [0], np.array([5]), np.array([[1.0, 0.0, 0.0]]), [])
    s = play_summary(w)
    assert "play/life_length" not in s["metrics"] and s["life len"] == "none ended"
    assert np.isclose(s["metrics"]["play/search_pi_entropy"], 0.0, atol=1e-6)
