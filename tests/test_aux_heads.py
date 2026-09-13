"""Tests for the auxiliary heads, focused on the properties that make them worth adding."""
import torch
import pytest

from src.models.world_on_rails.aux_heads import (
    TARGET_SPEEDS, TargetSpeedHead, AuxiliaryHeads, DepthHead, SemanticHead,
    two_hot_target_speed, target_speed_loss)


def test_two_hot_sums_to_one_and_is_unbiased():
    """The property that motivates two-hot over one-hot: the encoded distribution's expected
    value must reproduce the original continuous speed, including between bin centres."""
    bins = torch.tensor(TARGET_SPEEDS)
    speeds = torch.tensor([0.0, 2.0, 4.0, 9.0, 13.88888888, 18.5, 20.0])
    enc = two_hot_target_speed(speeds)
    assert enc.shape == (speeds.shape[0], len(TARGET_SPEEDS))
    assert torch.allclose(enc.sum(dim=-1), torch.ones(speeds.shape[0]), atol=1e-5)
    recovered = (enc * bins).sum(dim=-1)
    assert torch.allclose(recovered, speeds, atol=1e-4), f"biased: {recovered} vs {speeds}"


def test_two_hot_puts_all_mass_on_exact_bin_centres():
    """A speed sitting exactly on a bin should be one-hot, not smeared across neighbours."""
    enc = two_hot_target_speed(torch.tensor([8.0]))
    assert torch.isclose(enc.max(), torch.tensor(1.0), atol=1e-5)
    assert (enc > 1e-6).sum() == 1


def test_two_hot_handles_out_of_range_speeds():
    """Speeds beyond the table (a fast expert frame, or a corrupt measurement) must clamp rather
    than index out of bounds - this runs inside the training loop on real data."""
    enc = two_hot_target_speed(torch.tensor([-5.0, 999.0]))
    assert torch.allclose(enc.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert enc[0, 0].item() == pytest.approx(1.0, abs=1e-5)   # clamped to the 0.0 bin
    assert enc[1, -1].item() == pytest.approx(1.0, abs=1e-5)  # clamped to the top bin


def test_zero_speed_is_exactly_representable():
    """The whole reason this head exists: 'stopped' must be a class the policy can commit to,
    not a limit it approaches. Bin 0 is 0.0 m/s."""
    assert TARGET_SPEEDS[0] == 0.0
    enc = two_hot_target_speed(torch.tensor([0.0]))
    assert enc[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    head = TargetSpeedHead(in_dim=16)
    logits = torch.full((1, len(TARGET_SPEEDS)), -20.0)
    logits[0, 0] = 20.0
    assert head.expected_speed(logits).item() == pytest.approx(0.0, abs=1e-3)


def test_target_speed_loss_is_minimised_by_the_correct_bin():
    speeds = torch.tensor([8.0])
    good = torch.full((1, len(TARGET_SPEEDS)), -10.0)
    good[0, TARGET_SPEEDS.index(8.0)] = 10.0
    bad = torch.full((1, len(TARGET_SPEEDS)), -10.0)
    bad[0, -1] = 10.0
    assert target_speed_loss(good, speeds) < target_speed_loss(bad, speeds)


def test_expected_speed_recovers_bin_values():
    head = TargetSpeedHead(in_dim=8)
    for i, v in enumerate(TARGET_SPEEDS):
        logits = torch.full((1, len(TARGET_SPEEDS)), -30.0)
        logits[0, i] = 30.0
        assert head.expected_speed(logits).item() == pytest.approx(v, abs=1e-2)


def test_dense_heads_shapes_on_rectangular_features():
    """Feature maps are rectangular now (6x16 for a 192x512 input), not square - the heads must
    not assume otherwise."""
    feat = torch.randn(2, 64, 6, 16)
    d = DepthHead(64)(feat)
    s = SemanticHead(64, num_classes=7)(feat)
    assert d.shape[0] == 2 and d.shape[1] == 1
    assert s.shape[0] == 2 and s.shape[1] == 7
    assert d.shape[-2] * 2 > feat.shape[-2], "decoder did not upsample"
    assert d.min() >= 0.0 and d.max() <= 1.0, "depth must be normalised to [0,1]"


def test_dense_losses_resize_targets_to_predictions():
    feat = torch.randn(2, 64, 6, 16)
    depth_pred = DepthHead(64)(feat)
    sem_logits = SemanticHead(64, 7)(feat)
    # Full-resolution labels, deliberately a different size from the predictions.
    depth_t = torch.rand(2, 1, 192, 512)
    sem_t = torch.randint(0, 7, (2, 192, 512))
    assert torch.isfinite(DepthHead.loss(depth_pred, depth_t))
    assert torch.isfinite(SemanticHead.loss(sem_logits, sem_t))


def test_semantic_target_resize_does_not_invent_classes():
    """Nearest, never bilinear: interpolating integer class ids produces ids that do not exist
    and the loss would then be computed against a label nothing can predict."""
    feat = torch.randn(1, 32, 6, 16)
    logits = SemanticHead(32, num_classes=3)(feat)
    target = torch.zeros(1, 192, 512, dtype=torch.long)
    target[:, :, 256:] = 2  # only classes 0 and 2 present; bilinear would create class 1
    loss = SemanticHead.loss(logits, target)
    assert torch.isfinite(loss)


def test_aux_heads_disabled_by_default_and_skip_missing_targets():
    """Heads are opt-in, and a missing target must be skipped rather than trained against zeros."""
    aux = AuxiliaryHeads(feature_channels=64, state_dim=32)
    preds = aux(torch.randn(2, 64, 6, 16), torch.randn(2, 32))
    assert preds == {}

    aux = AuxiliaryHeads(feature_channels=64, state_dim=32,
                         use_target_speed=True, use_depth=True)
    preds = aux(torch.randn(2, 64, 6, 16), torch.randn(2, 32))
    assert set(preds) == {"target_speed_logits", "depth"}
    losses = aux.losses(preds, {"target_speed": torch.tensor([5.0, 12.0])})
    assert set(losses) == {"target_speed"}, "depth trained despite having no target"


def test_aux_loss_weights_are_applied():
    aux = AuxiliaryHeads(feature_channels=32, state_dim=16, use_target_speed=True)
    preds = aux(torch.randn(2, 32, 6, 16), torch.randn(2, 16))
    t = {"target_speed": torch.tensor([4.0, 8.0])}
    base = aux.losses(preds, t)["target_speed"]
    scaled = aux.losses(preds, t, weights={"target_speed": 0.25})["target_speed"]
    assert torch.isclose(scaled, base * 0.25, atol=1e-6)


def test_target_speed_head_gradients_flow():
    head = TargetSpeedHead(in_dim=16)
    logits = head(torch.randn(4, 16))
    target_speed_loss(logits, torch.tensor([0.0, 4.0, 10.0, 20.0])).backward()
    grads = [p.grad for p in head.net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
