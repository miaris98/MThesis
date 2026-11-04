"""Speed-ups for the live-encoder arm (TODO A54, user 2026-11-04): the s2d arm reads the stride-16 level, so the frozen encoder can stop
there (exact), and its convolutions can run channels_last (same values up to float rounding)."""
import pytest
import torch

timm = pytest.importorskip("timm")

from src.models.world_on_rails.carla_encoder import CarlaPretrainedEncoder


def _enc():
    torch.manual_seed(0)
    e = CarlaPretrainedEncoder(arch="regnety_032", expect_external_weights=True)
    return e.eval()


def test_forward_level_stops_at_stride_16_and_matches_the_full_pyramid():
    e = _enc()
    x = torch.rand(2, 3, 96, 256)
    full = e.forward_pyramid(x)
    lvl = e.forward_level(x, -2)
    assert e.feature_reductions[-2] == 16
    assert lvl.shape == full[-2].shape
    assert torch.equal(lvl, full[-2])          # exact: the same modules in the same order, the last stage never runs


def test_forward_level_skips_the_last_stage():
    e = _enc()
    last = list(e.encoder.keys())[-1]
    calls = []
    h = e.encoder[last].register_forward_hook(lambda *a: calls.append(1))
    e.forward_level(torch.rand(1, 3, 64, 128), -2)
    h.remove()
    assert calls == []


def test_channels_last_keeps_the_features():
    e = _enc()
    x = torch.rand(2, 3, 96, 256)
    ref = e.forward_level(x, -2)
    e.set_channels_last()
    out = e.forward_level(x, -2)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
