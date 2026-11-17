"""TODO A84: reason / danger heads on the policy token, their loss, the resume whitelist, the eval-time danger gate and the val readouts."""
import numpy as np
import pytest
import torch

from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
from src.models.world_on_rails.reason_heads import DangerGate, ReasonHeads, reason_loss, reason_term
from src.training.reason_metrics import ReasonValAccumulator, auroc, average_precision

KW = dict(backbone_name="resnet18", pretrained=False, model_size="10m", vision_grid=4, use_target_speed=True, route_points=4)


def _inputs(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(rgb=torch.rand(b, 3, 96, 256, generator=g), speed=torch.rand(b, 1, generator=g) * 10, command=torch.zeros(b, dtype=torch.long),
                route=torch.randn(b, 4, 2, generator=g))


def _batch(b=4):
    return {"reason_cause": torch.tensor([0, 1, 2, -1])[:b], "reason_dist": torch.tensor([0.0, 2.0, 1.5, 0.0])[:b],
            "reason_dist_valid": torch.tensor([0.0, 1.0, 1.0, 0.0])[:b], "reason_closing": torch.tensor([0.0, 5.0, 3.0, 0.0])[:b],
            "reason_closing_valid": torch.tensor([0.0, 1.0, 1.0, 0.0])[:b], "reason_brake": torch.tensor([0.0, 1.0, -1.0, 0.0])[:b],
            "reason_danger": torch.tensor([1.0, 0.0, -1.0, -1.0])[:b]}


def test_heads_shapes_and_detach_blocks_the_trunk_gradient():
    x = torch.randn(4, 32, requires_grad=True)
    out = ReasonHeads(32)(x)
    assert out["reason_cause_logits"].shape == (4, 6) and out["reason_danger_logit"].shape == (4,)
    loss, parts = reason_loss(out, _batch(), "cpu")
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert set(parts) == {"cause", "dist", "closing", "brake", "danger"}
    x2 = torch.randn(4, 32, requires_grad=True)
    loss2, _ = reason_loss(ReasonHeads(32, detach=True)(x2), _batch(), "cpu")
    loss2.backward()
    assert x2.grad is None                                                   # a classifier beside the policy: no gradient into the trunk


def test_loss_skips_unlabelled_frames_and_respects_the_weight():
    out = ReasonHeads(8)(torch.randn(4, 8))
    nolab = {k: (torch.full_like(v, -1) if k in ("reason_cause", "reason_brake", "reason_danger") else torch.zeros_like(v))
             for k, v in _batch().items()}
    assert reason_loss(out, nolab, "cpu")[0] is None
    assert reason_term(out, _batch(), 0.0, "cpu")[0] is None
    full, _ = reason_loss(out, _batch(), "cpu")
    t, _ = reason_term(out, _batch(), 0.25, "cpu")
    assert torch.allclose(t, 0.25 * full)


def test_policy_heads_are_opt_in_and_the_cascade_starts_at_the_base():
    assert "reason_danger_logit" not in QwenWorldOnRailsPolicy(**KW).eval()(**_inputs())
    torch.manual_seed(0)
    base = QwenWorldOnRailsPolicy(**KW).eval()
    torch.manual_seed(0)
    m = QwenWorldOnRailsPolicy(**KW, use_reason_heads=True, reason_cascade=True).eval()
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith(("reason_head.", "reason_embed."))}
    assert set(sd) == set(base.state_dict())                                 # only the new tensors are added ...
    assert all(torch.equal(sd[k], base.state_dict()[k]) for k in sd)         # ... and every other module keeps its init (forked RNG)
    a, b = base(**_inputs()), m(**_inputs())
    assert torch.allclose(a["target_speed_logits"], b["target_speed_logits"], atol=1e-6)   # zero-init cascade
    assert b["reason_cause_logits"].shape == (3, 6)
    with pytest.raises(ValueError):
        QwenWorldOnRailsPolicy(**{**KW, "use_target_speed": False}, use_reason_heads=True, reason_cascade=True)


def test_resume_from_a_checkpoint_without_the_heads_accepts_them_as_new():
    from src.training.wor_checkpoint import load_trainable_state
    base = QwenWorldOnRailsPolicy(**KW)
    m = QwenWorldOnRailsPolicy(**KW, use_reason_heads=True, reason_cascade=True)
    frozen = [k for k in m.state_dict() if k.startswith("encoder.")]
    state = {k: v for k, v in base.state_dict().items() if not k.startswith("encoder.")}
    changed = load_trainable_state(m, state, frozen, "E15")
    assert changed and all(k.startswith(("reason_head.", "reason_embed.")) for k in changed)


def test_danger_gate_closes_holds_releases_and_rearms(monkeypatch):
    g = DangerGate(theta=0.6, off=0.4, T=1.0, release=0.5, max_release=2, dt=0.25)
    assert g.update(0.2, 5.0, 8.0) == 8.0
    assert g.update(0.7, 5.0, 8.0) == 0.0                                    # closes at theta
    assert g.update(0.5, 2.0, 8.0) == 0.0                                    # hysteresis: still closed above `off`
    assert g.update(0.3, 1.0, 8.0) == 8.0                                    # opens below `off`
    for _ in range(4):                                                       # standing with the gate closed ...
        g.update(0.9, 0.0, 8.0)
    assert g.update(0.9, 0.0, 8.0) == 8.0                                    # ... T s later the stall breaker releases
    assert g.events["releases"] == 1
    for _ in range(3):
        g.update(0.9, 0.0, 8.0)
    for _ in range(6):
        g.update(0.9, 0.0, 8.0)
    assert g.n_release == 2 and g.update(0.9, 0.0, 8.0) == 8.0              # budget used: the gate gives up for this stop
    g.update(0.1, 4.0, 8.0)                                                  # driving again re-arms it
    assert g.n_release == 0 and g.update(0.9, 4.0, 8.0) == 0.0
    monkeypatch.delenv("WOR_DANGER_GATE", raising=False)
    assert DangerGate.from_env() is None
    monkeypatch.setenv("WOR_DANGER_GATE", "theta=0.7,off=0.3,T=2")
    assert DangerGate.from_env().theta == 0.7


def _confident(m):
    with torch.no_grad():
        m.reason_head.out.weight[-1].zero_()
        m.reason_head.out.bias[-1] = 5.0                                     # p(danger) ~ 0.99
        last = m.target_speed_head.net[-1]
        last.weight.zero_(); last.bias.zero_(); last.bias[4] = 10.0          # the speed head says 13.9 m/s
    return m


def test_act_records_danger_and_the_gate_stops_the_car(monkeypatch):
    kw = dict(backbone_name="resnet34", pretrained=False, route_points=4, model_size="10m", vision_grid=4, use_target_speed=True, use_reason_heads=True)
    rgb = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
    monkeypatch.delenv("WOR_DANGER_GATE", raising=False)
    m = _confident(QwenWorldOnRailsPolicy(**kw).eval())
    m.act(rgb=rgb, speed=5.0, command=3, device="cpu", route=np.zeros((4, 2), np.float32))
    assert m.last_debug["danger_p"] > 0.98 and m.last_debug["target_kmh"] > 40.0      # recorded, but no gate: the plain decode drives
    monkeypatch.setenv("WOR_DANGER_GATE", "theta=0.6")
    m2 = _confident(QwenWorldOnRailsPolicy(**kw).eval())
    _, throttle, brake = m2.act(rgb=rgb, speed=5.0, command=3, device="cpu", route=np.zeros((4, 2), np.float32))
    assert m2.last_debug["target_kmh"] == 0.0 and m2._danger_gate is not None and throttle == 0.0


def test_auroc_and_ap():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    assert abs(auroc(s, y) - 0.75) < 1e-9
    assert auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert np.isnan(auroc(s, np.zeros(4)))
    assert abs(average_precision(np.array([0.9, 0.1]), np.array([1, 0])) - 1.0) < 1e-9


def test_val_accumulator_reports_per_cause_speed_error_and_head_metrics():
    m = QwenWorldOnRailsPolicy(**KW, use_reason_heads=True).eval()
    acc = ReasonValAccumulator(m.target_speed_head)
    with torch.no_grad():
        out = m(**_inputs(4))
    batch = {**_batch(), "target_speed": torch.tensor([0.0, 5.0, 8.0, 3.0])}
    acc.add(out, batch, "cpu")
    r = acc.result()
    for k in ("val_reason_speed_err_none", "val_reason_speed_err_vehicle", "val_reason_pstop_danger", "val_reason_cause_acc", "val_reason_dist_mae_m"):
        assert k in r
    assert all(type(v) is float for v in r.values())          # numpy scalars in a checkpoint's metrics break torch>=2.6 weights_only loads
    assert ReasonValAccumulator(None).result() == {}
