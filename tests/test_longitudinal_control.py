"""The controller must act on the policy's longitudinal decision, not a constant.

`PIDController.__init__` sets `target_speed = 20.0` km/h and **nothing in this repository ever
writes to it**. `control_from_waypoints` falls back to that value whenever `target_speed_kmh` is
None, and `act()` never passed one - so every closed-loop evaluation this project has run
cruised at a fixed 20 km/h target, with the predicted waypoints contributing steering only.

That also makes the premise in TargetSpeedHead's docstring false: the PID did not "derive speed
from the spacing of predicted waypoints", it ignored them longitudinally. A policy that should
brake did not creep - it held 20 km/h.

These tests pin the fix and the fallback: with a target-speed head the controller must follow
its prediction, and without one the old constant behaviour must be preserved exactly, so
existing checkpoints evaluate as they always did.
"""
import pytest
import torch

from src.models.world_on_rails import QwenWorldOnRailsPolicy, WorldOnRailsPolicy
from src.models.world_on_rails.aux_heads import TARGET_SPEEDS

IMG = (192, 512)
# Torch tensors and route=None throughout: act() accepts either, and the numpy path cannot be
# exercised on a machine whose numpy and torch disagree about the array protocol. Longitudinal
# control is what is under test here; a zero route only removes steering intent.
RGB = torch.zeros(3, *IMG)
ROUTE = None


def _cnn(**kw):
    return WorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=4,
                              **kw).eval()


def _qwen(**kw):
    return QwenWorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=4,
                                  model_size="10m", vision_grid=(6, 16), **kw).eval()


def _force_bin(policy, index):
    """Pins the head's output to one speed bin, so the test asserts on control rather than on
    whatever an untrained head happens to predict."""
    head = policy.target_speed_head
    with torch.no_grad():
        head.net[-1].weight.zero_()
        head.net[-1].bias.zero_()
        head.net[-1].bias[index] = 40.0     # saturates the softmax onto this bin
    return TARGET_SPEEDS[index]


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_predicted_stop_brakes_instead_of_cruising(build):
    """Bin 0 is exactly 0.0 m/s. A confident stop must produce brake and no throttle - the
    behaviour the constant 20 km/h target made impossible."""
    policy = build(use_target_speed=True)
    _force_bin(policy, 0)
    steer, throttle, brake = policy.act(rgb=RGB, speed=8.0, command=3, device="cpu", route=ROUTE)
    assert throttle == 0.0 and brake > 0.0, (throttle, brake)


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_predicted_cruise_accelerates_when_below_target(build):
    """Top bin is 20 m/s = 72 km/h. A car doing 18 km/h must accelerate, where the old constant
    target of 20 km/h would have held it nearly still."""
    policy = build(use_target_speed=True)
    _force_bin(policy, len(TARGET_SPEEDS) - 1)
    steer, throttle, brake = policy.act(rgb=RGB, speed=5.0, command=3, device="cpu", route=ROUTE)
    assert throttle > 0.0 and brake == 0.0, (throttle, brake)


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_the_controller_target_is_published_in_kmh(build):
    """eval_wor.py's HUD reads `controller.target_speed`. It must show the live decision, in the
    same km/h the PID works in - not the never-updated default."""
    policy = build(use_target_speed=True)
    mps = _force_bin(policy, 3)
    policy.act(rgb=RGB, speed=4.0, command=3, device="cpu", route=ROUTE)
    assert policy.controller.target_speed == pytest.approx(mps * 3.6, rel=1e-3)


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_without_a_head_the_old_constant_is_preserved(build):
    """Checkpoints trained without the head must evaluate exactly as before."""
    policy = build(use_target_speed=False)
    assert policy.target_speed_head is None
    before = policy.controller.target_speed
    policy.act(rgb=RGB, speed=4.0, command=3, device="cpu", route=ROUTE)
    assert policy.controller.target_speed == before == 20.0


@pytest.mark.parametrize("build", [_cnn, _qwen], ids=["cnn", "qwen"])
def test_longitudinal_control_now_depends_on_the_image(build):
    """End to end: the two fixes together. The head reads the trunk (which has seen the
    image) and the controller reads the head, so the speed the car is aiming for must now
    depend on the pixels. Before either fix this was a constant, whatever was in frame.

    Asserted on the controller target rather than on throttle: throttle clips to [0, 1], and
    an untrained head predicts a target far above the test speed, so both images saturate at
    1.0 and the difference is hidden. That the controller acts on the target is pinned by the
    forced-bin tests above; this pins that the target follows the image."""
    policy = build(use_target_speed=True, target_speed_input="policy")
    g = torch.Generator().manual_seed(11)
    a = torch.rand(3, *IMG, generator=g)
    b = torch.rand(3, *IMG, generator=g)
    policy.act(rgb=a, speed=6.0, command=3, device="cpu", route=ROUTE)
    target_a = policy.controller.target_speed
    policy.act(rgb=b, speed=6.0, command=3, device="cpu", route=ROUTE)
    target_b = policy.controller.target_speed
    assert target_a != target_b, (
        f"the car aims for {target_a} km/h regardless of what it sees")
