"""TODO A28: inference-time read-out of the target-speed bins (mean / median / quantile)."""
import math

import pytest
import torch

from src.models.world_on_rails.aux_heads import TARGET_SPEEDS, TargetSpeedHead, parse_speed_decode, speed_quantile

BINS = torch.tensor(TARGET_SPEEDS)


def _logits(probs):
    p = torch.tensor(probs, dtype=torch.float32)
    return torch.log(p.clamp_min(1e-12)).unsqueeze(0)


def _dist(**mass):
    p = [0.0] * len(TARGET_SPEEDS)
    for speed, m in mass.items():
        p[TARGET_SPEEDS.index(float(speed[1:]))] = m
    return p


def test_parse():
    assert parse_speed_decode("mean") is None
    assert parse_speed_decode("median") == 0.5
    assert parse_speed_decode("quantile:0.3") == pytest.approx(0.3)
    for bad in ("quantile:0", "quantile:1.5", "mode"):
        with pytest.raises(ValueError):
            parse_speed_decode(bad)


def test_default_mean_is_unchanged(monkeypatch):
    monkeypatch.delenv("WOR_SPEED_DECODE", raising=False)
    head = TargetSpeedHead(4)
    logits = torch.randn(3, len(TARGET_SPEEDS))
    ref = (logits.softmax(dim=-1) * BINS).sum(dim=-1)
    assert torch.equal(head.expected_speed(logits), ref)


def test_stop_go_split_mean_creeps_median_does_not(monkeypatch):
    # the TODO's example: P(stop) = P(8 m/s) = 0.5
    logits = _logits(_dist(v0=0.5, v8=0.5))
    monkeypatch.setenv("WOR_SPEED_DECODE", "mean")
    assert TargetSpeedHead(4).expected_speed(logits).item() == pytest.approx(4.0, abs=1e-4)
    monkeypatch.setenv("WOR_SPEED_DECODE", "median")
    assert TargetSpeedHead(4).expected_speed(logits).item() == 0.0      # a tie resolves to the slower mode


def test_median_snaps_to_majority_mode(monkeypatch):
    monkeypatch.setenv("WOR_SPEED_DECODE", "median")
    head = TargetSpeedHead(4)
    assert head.expected_speed(_logits(_dist(v0=0.4, v8=0.6))).item() == 8.0
    assert head.expected_speed(_logits(_dist(v0=0.6, v8=0.4))).item() == 0.0


def test_low_quantile_is_more_cautious():
    probs = torch.tensor([_dist(v0=0.2, v8=0.3, v10=0.5)])
    assert speed_quantile(probs, BINS, 0.5).item() == 8.0
    assert speed_quantile(probs, BINS, 0.15).item() == 0.0
    assert speed_quantile(probs, BINS, 0.9).item() == 10.0


def test_explicit_decode_overrides_env(monkeypatch):
    monkeypatch.setenv("WOR_SPEED_DECODE", "median")
    head = TargetSpeedHead(4, decode="mean")
    assert head.decode is None
    assert math.isclose(head.expected_speed(_logits(_dist(v0=0.5, v8=0.5))).item(), 4.0, abs_tol=1e-4)
