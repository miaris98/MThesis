"""The target-speed head must be able to see the road.

Both arms used to feed it ego state only - speed, command and route - so it was structurally
incapable of the task it is named for: no red light, no lead vehicle, no pedestrian reaches it,
and the best it can learn is "slow down for curvy routes". Its gradient still flowed into the
shared state projections, so it was not merely useless but actively pulling on embeddings the
trunk depends on.

These tests pin the property that matters and cannot be checked by reading shapes: change only
the image, and the predicted target speed must change.
"""
import pytest
import torch

from src.models.world_on_rails import QwenWorldOnRailsPolicy, WorldOnRailsPolicy
from src.models.world_on_rails.ray_geometry import (
    RAY_GEOMETRY_CHANNELS, vision_grid_geometry)
from tests._sanity import soft_check

torch.manual_seed(0)
IMG = (192, 512)


def _cnn(**kw):
    return WorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, freeze_backbone=True,
                              route_points=4, use_target_speed=True, **kw).eval()


def _qwen(**kw):
    return QwenWorldOnRailsPolicy(backbone_name="resnet34", pretrained=False,
                                  freeze_backbone=True, route_points=4, model_size="10m",
                                  vision_grid=(6, 16), use_target_speed=True, **kw).eval()


def _state(b=2):
    return (torch.tensor([[5.0]] * b), torch.tensor([3] * b), torch.zeros(b, 4, 2))


def _two_different_images(b=2):
    g = torch.Generator().manual_seed(7)
    a = torch.rand(b, 3, *IMG, generator=g)
    return a, torch.rand(b, 3, *IMG, generator=g)


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_target_speed_responds_to_the_image(build):
    """The whole point. Same ego state, different pixels -> different target-speed logits."""
    policy = build(target_speed_input="policy")
    speed, cmd, route = _state()
    rgb_a, rgb_b = _two_different_images()
    with torch.no_grad():
        a = policy(rgb_a, speed, cmd, route)["target_speed_logits"]
        b = policy(rgb_b, speed, cmd, route)["target_speed_logits"]
    assert not torch.allclose(a, b, atol=1e-6), (
        "target-speed logits are identical for two different images - the head is not "
        "receiving vision")
    # A hard != is satisfied by a difference of 1e-7, which would still mean the head is only
    # nominally looking. At init, with random weights, a meaningful swing in the input should
    # produce more than a rounding-error-sized swing in the logits - if it doesn't, that's not
    # wrong enough to fail the build over, but it is worth a second look (e.g. a near-zero-init
    # layer swallowing the signal, or an accidental detach on part of the path).
    max_logit_delta = (a - b).abs().max().item()
    soft_check(max_logit_delta > 1e-3,
               f"target-speed logits barely moved between two random images (max |delta| = "
               f"{max_logit_delta:.2e}) - technically != but close enough to blind that it's "
               f"worth checking nothing upstream is nearly detached or zero-initialized")


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_state_mode_reproduces_the_old_blind_behaviour(build):
    """The legacy path is kept so earlier runs stay reproducible; pin that it *is* the old
    behaviour, i.e. still blind. If this ever starts failing, the two modes have converged and
    the ablation no longer means anything."""
    policy = build(target_speed_input="state")
    speed, cmd, route = _state()
    rgb_a, rgb_b = _two_different_images()
    with torch.no_grad():
        a = policy(rgb_a, speed, cmd, route)["target_speed_logits"]
        b = policy(rgb_b, speed, cmd, route)["target_speed_logits"]
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_target_speed_gradient_reaches_the_trunk_not_the_frozen_encoder(build):
    """Vision reaching the head means its loss now shapes the trainable trunk - which is the
    benefit. It must still not reach the frozen backbone, which is what makes the perception
    losses in TF++ inert for this project."""
    policy = build(target_speed_input="policy").train()
    speed, cmd, route = _state()
    rgb = torch.rand(2, 3, *IMG)
    policy(rgb, speed, cmd, route)["target_speed_logits"].square().mean().backward()

    trunk_grads = [p.grad for n, p in policy.named_parameters()
                   if p.grad is not None and not n.startswith("encoder.")]
    assert trunk_grads, "no trainable parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in trunk_grads)
    assert all(p.grad is None for n, p in policy.named_parameters() if n.startswith("encoder."))


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_rail_heads_can_be_switched_off(build):
    """With q_loss_weight=0 and PDM-Lite's all-zero target_q these heads train on nothing and
    act() never reads them. Off, they must vanish from the output rather than emit zeros."""
    on, off = build(use_rail_q=True), build(use_rail_q=False)
    speed, cmd, route = _state()
    rgb = torch.rand(2, 3, *IMG)
    with torch.no_grad():
        out_on, out_off = on(rgb, speed, cmd, route), off(rgb, speed, cmd, route)
    assert "selected_rail_q" in out_on and "rail_q" in out_on
    assert "selected_rail_q" not in out_off and "rail_q" not in out_off
    assert "q_map" not in out_off
    # The outputs that drive the car are unaffected.
    assert out_off["selected_waypoints"].shape == out_on["selected_waypoints"].shape
    assert not any(n.startswith(("rail_head", "q_head.rail_head", "q_head.q_map_head"))
                   for n, _ in off.named_parameters())


def test_loss_tolerates_a_policy_without_rail_heads():
    from src.training.wor_eval import waypoint_losses
    policy = _qwen(use_rail_q=False)
    speed, cmd, route = _state()
    with torch.no_grad():
        out = policy(torch.rand(2, 3, *IMG), speed, cmd, route)
    losses = waypoint_losses(out, torch.zeros(2, 5, 2), torch.zeros(2, 9), q_loss_weight=0.0)
    assert torch.isfinite(losses["total"]) and float(losses["q"]) == 0.0


# --- ray geometry -------------------------------------------------------------------------

def test_ray_geometry_shape_and_bounds():
    geo = vision_grid_geometry((6, 16), crop_bottom_frac=0.25)
    assert geo.shape == (RAY_GEOMETRY_CHANNELS, 6, 16)
    assert torch.isfinite(geo).all()
    assert geo.abs().max() <= 1.0 + 1e-6, "every channel is a sin/cos or a normalised distance"


def test_azimuth_increases_left_to_right():
    """Channel 0 is sin(azimuth); it must rise monotonically across columns, which is what
    makes it usable as a bearing. Also pins the sign convention: +x is to the right."""
    geo = vision_grid_geometry((6, 16), crop_bottom_frac=0.25)
    az = geo[0, 0]
    assert torch.all(az[1:] > az[:-1])
    assert az[0] < 0 < az[-1]


def test_ground_plane_is_valid_only_below_the_horizon_and_gets_nearer_downward():
    """With pitch=0 the horizon is the vertical centre of the *uncropped* frame, so after the
    bottom-quarter crop only the lower rows see road. Rows further down must map nearer."""
    geo = vision_grid_geometry((6, 16), crop_bottom_frac=0.25)
    valid, fwd = geo[6], geo[4]
    assert valid[0].sum() == 0, "top row is sky - no ground intersection"
    assert valid[-1].sum() > 0, "bottom row must see road"
    centre = fwd[:, 8]
    rows = [r for r in range(6) if valid[r, 8] > 0]
    assert len(rows) >= 2
    assert all(centre[rows[i]] > centre[rows[i + 1]] for i in range(len(rows) - 1))
    # The docstring's own math says ~2 of 6 rows should see ground at this crop. len(rows) >= 2
    # above is the hard floor (the geometry would still be usable with more), but a grid that
    # somehow sees ground everywhere or on a single row would mean the flat-ground/horizon
    # assumption drifted from what was validated - a modelling regression, not a shape bug.
    soft_check(2 <= len(rows) <= 3,
               f"expected ~2-3 of 6 rows to have a valid ground intersection at "
               f"crop_bottom_frac=0.25 (matches the documented horizon-at-2/3 math), got "
               f"{len(rows)} - re-check the camera pitch/crop assumption if this changed on "
               f"purpose")


def test_geometry_is_independent_of_resize_and_is_cached():
    """It is a function of the camera and the grid, not of --img_size: scaling an image does
    not change where its pixels point. Caching matters because it is rebuilt every forward."""
    a = vision_grid_geometry((6, 16), crop_bottom_frac=0.25)
    b = vision_grid_geometry((6, 16), crop_bottom_frac=0.25)
    assert a is b
    assert not torch.equal(a, vision_grid_geometry((6, 16), crop_bottom_frac=0.0))


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_ray_geometry_widens_the_input_and_changes_predictions(build):
    plain, geom = build(use_ray_geometry=False), build(use_ray_geometry=True)
    speed, cmd, route = _state()
    rgb = torch.rand(2, 3, *IMG)
    with torch.no_grad():
        a = plain(rgb, speed, cmd, route)["selected_waypoints"]
        b = geom(rgb, speed, cmd, route)["selected_waypoints"]
    assert a.shape == b.shape
    assert not torch.allclose(a, b), "geometry channels reached no parameter"


def test_globally_pooled_grid_has_no_meaningful_geometry():
    """One token covering the whole frame has no per-cell bearing; it must degrade to zeros
    with the valid flag clear rather than inventing a direction."""
    geo = vision_grid_geometry((1, 1), crop_bottom_frac=0.25)
    assert torch.count_nonzero(geo) == 0
