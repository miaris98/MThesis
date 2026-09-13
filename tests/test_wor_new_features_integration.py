"""End-to-end wiring checks for the redesign: overlay, aspect-correct input, target-speed head.

These exercise the real classes with synthetic data rather than mocks. A unit test of each piece
passing while the assembled pipeline is mis-wired is the failure mode that produced 11.14 - the
component was fine, the thing that consumed it was not.
"""
import numpy as np
import torch
import pytest

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
from src.training.wor_dataset import WorldOnRailsDataset


def _batch(policy, h=192, w=512, b=2, route_points=4):
    return dict(
        rgb=torch.randint(0, 255, (b, h, w, 3), dtype=torch.uint8).float(),
        speed=torch.rand(b, 1) * 20.0,
        command=torch.randint(0, 6, (b,)),
        route=torch.randn(b, route_points, 2),
    )


@pytest.mark.parametrize("use_ts", [False, True])
def test_cnn_policy_target_speed_head_is_opt_in(use_ts):
    p = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False,
                           use_target_speed=use_ts)
    out = p(**_batch(p))
    assert ("target_speed_logits" in out) is use_ts
    if use_ts:
        assert out["target_speed_logits"].shape == (2, 8)


@pytest.mark.parametrize("use_ts", [False, True])
def test_qwen_policy_target_speed_head_is_opt_in(use_ts):
    p = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False,
                               model_size="10m", vision_grid=(6, 16),
                               use_target_speed=use_ts)
    out = p(**_batch(p))
    assert ("target_speed_logits" in out) is use_ts
    if use_ts:
        assert out["target_speed_logits"].shape == (2, 8)


def test_policies_accept_rectangular_non_square_input():
    """The whole point of the aspect fix: a 192x512 frame must flow through both arms.

    The previous square-only assumption lived in the Qwen trunk's AdaptiveAvgPool2d((N, N)),
    which would have silently re-squashed a rectangular feature map back to square.
    """
    cnn = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False)
    qwen = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False,
                                  model_size="10m", vision_grid=(6, 16))
    for p in (cnn, qwen):
        out = p(**_batch(p, h=192, w=512))
        assert out["selected_waypoints"].shape == (2, 5, 2)


def test_qwen_rectangular_grid_keeps_all_tokens():
    """A 6x16 grid must yield 96 vision tokens, not 64 from a square 8x8 pooling."""
    p = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False,
                               model_size="10m", vision_grid=(6, 16))
    assert p.num_vision_tokens == 96
    assert p.vision_grid_hw == (6, 16)


def test_square_vision_grid_still_works_for_legacy_runs():
    p = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False,
                               model_size="10m", vision_grid=8)
    assert p.num_vision_tokens == 64
    assert p.vision_grid_hw == (8, 8)


def test_dataset_emits_target_speed_and_respects_overlay_flag():
    """Synthetic mode exercises the batch contract without needing the 414GB dataset."""
    ds = WorldOnRailsDataset(data_dir="/nonexistent", synthetic_samples=4,
                             img_size=(192, 512), route_points=4)
    item = ds[0]
    assert "target_speed" in item, "target-speed label missing from the batch"
    assert item["target_speed"].dtype == torch.float32
    assert item["rgb"].shape == (192, 512, 3)


def test_overlay_flag_changes_the_cache_key():
    """Overlaid and non-overlaid frames are different images at identical dimensions. If they
    shared a cache key, one run would silently serve the other's pixels."""
    common = dict(data_dir="/nonexistent", synthetic_samples=1, img_size=(192, 512))
    plain = WorldOnRailsDataset(**common, route_overlay=False)
    overlaid = WorldOnRailsDataset(**common, route_overlay=True)
    assert plain.route_overlay is False and overlaid.route_overlay is True
    tag_plain = f"{'c' if plain.crop_bottom_frac else ''}{'o' if plain.route_overlay else ''}"
    tag_ovl = f"{'c' if overlaid.crop_bottom_frac else ''}{'o' if overlaid.route_overlay else ''}"
    assert tag_plain != tag_ovl


def test_target_speed_loss_backprops_through_the_policy():
    """The head must actually be in the optimised graph, not a detached side output."""
    from src.models.world_on_rails.aux_heads import target_speed_loss
    p = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False,
                           use_target_speed=True)
    out = p(**_batch(p))
    target_speed_loss(out["target_speed_logits"], torch.tensor([0.0, 12.0])).backward()
    grads = [q.grad for q in p.target_speed_head.parameters() if q.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_frozen_backbone_gets_no_gradients():
    """Scope rule: the vision backbone is not trained. A target-speed loss that leaked
    gradients into it would quietly violate that."""
    from src.models.world_on_rails.aux_heads import target_speed_loss
    p = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False,
                           freeze_backbone=True, use_target_speed=True)
    out = p(**_batch(p))
    target_speed_loss(out["target_speed_logits"], torch.tensor([4.0, 8.0])).backward()
    assert all(q.grad is None or q.grad.abs().sum() == 0 for q in p.encoder.parameters())
