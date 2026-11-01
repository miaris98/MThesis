"""TODO A71: scripts/analysis/qhead_offline_check.py - metrics of a trained Q-critic head against the counterfactual labels, overall and on the
frames where the expert bypasses (the frames the 2026-11-01 closed-loop smoke failed on: the car stopped behind the obstacle instead of passing)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("qhead_offline_check", Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "qhead_offline_check.py")
qc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qc)

NV, ND = 8, 5


def _q(best_v, best_d, base=0.0):
    q = np.full((NV, ND), base, np.float32)
    q[best_v, best_d] = base + 5.0
    return q


def test_perfect_head_scores_one_everywhere():
    tgt = np.stack([_q(3, 2), _q(0, 0), _q(5, 4)])
    tgt += np.random.default_rng(0).normal(0, 0.1, tgt.shape).astype(np.float32)   # distinct values, so ranks are defined
    m = qc.qhead_metrics(tgt.copy(), tgt, np.array([0, -1, 1]))
    assert m["frames"] == 3 and m["mean_abs_error"] == pytest.approx(0.0)
    assert m["mean_spearman_rho"] == pytest.approx(1.0) and m["argmax_in_label_top1_share"] == 1.0
    assert m["bypass_frames"] == 2 and m["bypass_label_expert_side_share"] == 1.0 and m["bypass_pred_expert_side_share"] == 1.0


def test_head_that_stays_in_lane_fails_the_bypass_metric_but_not_the_overall_lateral_one():
    """The smoke head's lateral agreement (95.5%) was dominated by in-lane frames; on bypass frames a head that never leaves the lane scores 0."""
    tgt = np.stack([_q(3, 2)] * 18 + [_q(2, 0), _q(2, 0)])
    pred = np.stack([_q(3, 2)] * 20)
    side = np.array([0] * 18 + [-1, -1])
    m = qc.qhead_metrics(pred, tgt, side)
    assert m["lat_argmax_agreement"] == pytest.approx(0.9)
    assert m["bypass_frames"] == 2 and m["bypass_label_expert_side_share"] == 1.0 and m["bypass_pred_expert_side_share"] == 0.0


def test_no_bypass_frames_gives_none_not_zero():
    tgt = np.stack([_q(3, 2)])
    m = qc.qhead_metrics(tgt, tgt, np.array([0]))
    assert m["bypass_frames"] == 0 and m["bypass_pred_expert_side_share"] is None


def test_expert_side_follows_the_route_shift():
    ro = [[2.5 + j, 0.0] for j in range(20)]
    assert qc.expert_side({"target_speed": 8.0, "route_original": ro, "route": [[2.5 + j, -3.5 if j > 3 else 0.0] for j in range(20)]}) == -1
    assert qc.expert_side({"target_speed": 8.0, "route_original": ro, "route": ro}) == 0
    assert qc.expert_side({"target_speed": 8.0}) == 0


def test_measurement_path_of_an_rgb_frame():
    assert Path(qc.measurement_path("/d/Town12/data/X/Route1_Rep0/rgb/0042.jpg")) == Path("/d/Town12/data/X/Route1_Rep0/measurements/0042.json.gz")


def test_tied_label_columns_do_not_count_as_the_experts_side():
    """Review: np.argmax picks the leftmost of tied columns, so left bypasses were over-counted for the labels (not for a continuous head)."""
    tgt = np.zeros((1, NV, ND), np.float32)                     # every action ties: the label does not prefer the expert's (left) side
    m = qc.qhead_metrics(tgt.copy(), tgt, np.array([-1]))
    assert m["bypass_label_expert_side_share"] == 0.0


def test_decode_rule_calibration_over_lat_margin():
    """The decode rule's lane choice depends on Q only (column values against lat_margin), so it can be calibrated offline: how often it leaves the
    lane on in-lane frames, and how often it takes the expert's side on bypass frames."""
    inlane = _q(3, 2)
    inlane[:, 4] = inlane[:, 2].max() + 0.6                     # a noisy head: the right column beats the lane by 0.6
    bypass = _q(2, 0)                                           # the left column beats the lane by 5
    m = qc.qhead_metrics(np.stack([inlane, bypass]), np.stack([_q(3, 2), _q(2, 0)]), np.array([0, -1]), margins=(0.5, 1.0))
    assert m["decode_leave_inlane_m0p5"] == 1.0 and m["decode_leave_inlane_m1"] == 0.0
    assert m["decode_expert_side_bypass_m0p5"] == 1.0 and m["decode_expert_side_bypass_m1"] == 1.0
