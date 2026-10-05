"""TODO A50 / A57 / A52: the eval-time safety layer. Pure python, no CARLA."""
import numpy as np

from src.agents.safety_layer import SafetyLayer, quantile_speed

BINS = np.array([0.0, 4.0, 8.0, 10.0, 13.9, 16.0, 17.8, 20.0])


def P(**mass):
    p = np.zeros(8)
    for k, v in mass.items():
        p[int(k[1:])] = v
    return p


def test_quantile_matches_the_torch_rule_and_is_not_interpolated():
    p = P(b0=0.4, b2=0.6)
    assert quantile_speed(p, BINS, 0.3) == 0.0 and quantile_speed(p, BINS, 0.5) == 8.0 and quantile_speed(p, BINS, 0.999) == 8.0


def test_no_env_means_no_layer(monkeypatch):
    for k in ("WOR_NEWSVENDOR", "WOR_CREEP_CLAMP", "WOR_CONTACT_REFLEX", "WOR_SPEED_SCALE"):
        monkeypatch.delenv(k, raising=False)
    assert SafetyLayer.from_env() is None


def test_speed_scale_control_scales_the_mean_only():
    L = SafetyLayer(speed_scale=0.8)
    t, brake = L.update(10.0, P(b2=1.0), BINS, 8.0, 0.0)
    assert abs(t - 6.4) < 1e-9 and not brake
    assert L.update(10.0, P(b0=1.0), BINS, 0.0, 0.0)[0] == 0.0


def test_newsvendor_reads_a_low_quantile_then_relaxes_to_the_mean():
    L = SafetyLayer(newsvendor={"q0": 0.3, "T": 8.0, "qmax": 0.7}, dt=0.05)
    p = P(b0=0.4, b2=0.6)                      # mean 4.8, quantile 0.3 -> 0, quantile 0.7 -> 8
    assert L.update(10.0, p, BINS, 4.8, 0.0)[0] == 0.0          # moving: cautious low quantile
    for _ in range(int(8.0 / 0.05) + 5):                         # standing for > T
        t, _ = L.update(0.5, p, BINS, 4.8, 0.0)
    assert abs(t - 4.8) < 1e-9                                   # after T: the plain mean decode
    t_mid = None
    L2 = SafetyLayer(newsvendor={"q0": 0.3, "T": 8.0, "qmax": 0.7}, dt=0.05)
    for _ in range(int(5.0 / 0.05)):
        t_mid, _ = L2.update(0.5, p, BINS, 4.8, 0.0)
    assert t_mid == 8.0                                           # quantile has risen above the stop mass (0.4): go


def test_creep_clamp_blocks_a_bimodal_creep_and_the_stall_breaker_releases_it():
    L = SafetyLayer(creep_clamp=True, dt=0.05)
    bimodal = P(b0=0.5, b2=0.5)
    out = [L.update(1.0, bimodal, BINS, 4.0, 0.0)[0] for _ in range(int(3.0 / 0.05))]
    assert all(v == 0.0 for v in out)                             # mean 4 m/s creep suppressed
    for _ in range(int(2.0 / 0.05)):                              # standing > 4 s, stop mass 0.5 < 0.9
        t = L.update(0.0, bimodal, BINS, 4.0, 0.0)[0]
    assert t == 4.0                                               # released: the mean again
    assert L.update(0.0, bimodal, BINS, 4.0, 0.0)[0] == 4.0       # stays released while it keeps standing
    # a confident stop stays a stop at any standing time
    L = SafetyLayer(creep_clamp=True, dt=0.05)
    for _ in range(int(10.0 / 0.05)):
        t = L.update(0.0, P(b0=0.95, b2=0.05), BINS, 0.4, 0.0)[0]
    assert t == 0.0


def test_clamp_does_not_touch_normal_driving():
    L = SafetyLayer(creep_clamp=True, contact_reflex=True, dt=0.05)
    for _ in range(100):
        t, brake = L.update(12.0, P(b0=0.4, b4=0.6), BINS, 8.3, 0.0)
    assert t == 8.3 and not brake


def test_contact_reflex_fires_on_a_speed_drop_without_braking_and_holds_three_seconds():
    L = SafetyLayer(contact_reflex=True, dt=0.05)
    for _ in range(40):
        L.update(8.0, P(b2=1.0), BINS, 8.0, 0.0)
    L.update(8.0, P(b2=1.0), BINS, 8.0, 0.0)
    t, brake = L.update(4.5, P(b2=1.0), BINS, 8.0, 0.0)           # 3.5 m/s drop in one tick, commanded brake 0
    assert brake and t == 0.0 and L.events["contacts"] == 1
    held = sum(1 for _ in range(int(2.8 / 0.05)) if L.update(0.0, P(b2=1.0), BINS, 8.0, 0.0)[1])
    assert held >= 54                                             # still holding at ~2.8 s
    for _ in range(int(0.5 / 0.05)):
        _, brake = L.update(0.0, P(b2=1.0), BINS, 8.0, 0.0)
    assert not brake                                              # hold over; the clamp (stop mass 0) now lets the car move once the breaker allows


def test_a_hard_brake_by_the_policy_is_not_a_contact():
    L = SafetyLayer(contact_reflex=True, dt=0.05)
    for _ in range(40):
        L.update(8.0, P(b2=1.0), BINS, 8.0, 0.0)
    _, brake = L.update(4.5, P(b2=1.0), BINS, 0.0, 1.0)           # commanded brake 1.0: expected deceleration
    assert not brake and L.events["contacts"] == 0


def test_policy_act_publishes_debug_and_honours_the_layer(monkeypatch):
    """End to end through QwenWorldOnRailsPolicy.act on CPU: the layer is built from the environment, last_debug carries the posterior, and with a clamp
    the commanded target speed is 0 when the stop mass is high."""
    import torch
    from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
    monkeypatch.setenv("WOR_CREEP_CLAMP", "1")
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=4, model_size="10m", vision_grid=4, use_target_speed=True).eval()
    with torch.no_grad():  # force a bimodal stop / go posterior: bin 0 and bin 2 equal
        last = m.target_speed_head.net[-1]
        last.weight.zero_(); last.bias.zero_(); last.bias[0] = 5.0; last.bias[2] = 5.0
    rgb = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
    steer, throttle, brake = m.act(rgb=rgb, speed=1.0, command=3, device="cpu", route=np.zeros((4, 2), np.float32))
    assert m.last_debug["probs"].shape == (8,) and abs(m.last_debug["probs"][0] - 0.5) < 0.01
    assert m.last_debug["target_kmh"] == 0.0                      # clamp: stop mass 0.5 >= 0.3 at 1 m/s
    assert m._safety is not None and m._safety.events["clamp_ticks"] == 1
