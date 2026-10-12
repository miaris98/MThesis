"""Pins the speed-unit contract through the inference path.

`act()` feeds one scalar to two consumers that want different units: the network, which
trained on PDM-Lite's `speed` field in m/s, and the PID, whose error term is
`(target_speed_kmh - current_speed_kmh) / 3.6`. For most of this project's life it
handed the same number to both, so whichever caller was right for one was wrong for the
other - and the two callers in the repo disagreed with each other, so both mistakes were
live simultaneously. Neither raises.

These tests fix the convention: **everything upstream of `act()` is metres per second**.
"""
import numpy as np
import pytest
import torch

from src.models.world_on_rails.pid_controller import PIDController


class _SpyController(PIDController):
    """Records what the policy handed the controller, and returns a fixed control."""

    def __init__(self):
        super().__init__()
        self.seen_speed_kmh = None

    def control_from_waypoints(self, waypoints, current_speed_kmh, target_speed_kmh=None):
        self.seen_speed_kmh = current_speed_kmh
        return 0.0, 0.0, 0.0


def _policy():
    from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
    net = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False)
    net.controller = _SpyController()
    return net


def test_act_converts_metres_per_second_to_kmh_for_the_controller():
    """20 m/s is 72 km/h. The controller must see 72, not 20 - at 20 it would read the
    car as far below the 20 km/h target and hold full throttle all the way to 72 km/h."""
    net = _policy()
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    net.act(rgb=rgb, speed=20.0, command=3, device="cpu",
            route=np.zeros((4, 2), dtype=np.float32))

    assert net.controller.seen_speed_kmh == pytest.approx(72.0)


def test_act_feeds_the_network_the_unconverted_metres_per_second():
    """The other half of the same contract: the value reaching `speed_mlp` must be the
    m/s the dataset stores, not the km/h the controller needs."""
    net = _policy()
    captured = {}
    original = net.embed_state

    def spy(speed, command, route=None):
        captured["speed"] = float(speed.view(-1)[0].item())
        return original(speed, command, route)

    net.embed_state = spy
    net.act(rgb=np.zeros((256, 256, 3), dtype=np.uint8), speed=20.0, command=3,
            device="cpu", route=np.zeros((4, 2), dtype=np.float32))

    assert captured["speed"] == pytest.approx(20.0)


def test_a_stationary_car_is_zero_in_both_units():
    net = _policy()
    net.act(rgb=np.zeros((256, 256, 3), dtype=np.uint8), speed=0.0, command=3,
            device="cpu", route=np.zeros((4, 2), dtype=np.float32))
    assert net.controller.seen_speed_kmh == pytest.approx(0.0)


def test_agent_treats_the_speedometer_dict_and_a_bare_float_as_the_same_unit():
    """CARLA's speedometer pseudo-sensor reports {'speed': m/s} while the eval scripts
    pass a bare float. Those two shapes previously meant *different units* - the dict
    was multiplied by 3.6 and the float was not - so the same physical speed produced
    two different network inputs depending only on how the caller packed it."""
    from src.agents.wor_agent import WorldOnRailsAgent

    agent = WorldOnRailsAgent.__new__(WorldOnRailsAgent)  # skip weight loading
    agent.device = "cpu"
    agent.step_counter = 0
    agent.net = _policy()

    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    route = np.zeros((4, 2), dtype=np.float32)

    agent.run_step({"rgb_front": (0, rgb), "speed": (0, 10.0),
                    "command": 3, "route": route})
    from_float = agent.net.controller.seen_speed_kmh

    agent.run_step({"rgb_front": (0, rgb), "speed": (0, {"speed": 10.0}),
                    "command": 3, "route": route})
    from_dict = agent.net.controller.seen_speed_kmh

    assert from_float == pytest.approx(from_dict)
    assert from_float == pytest.approx(36.0)  # 10 m/s
