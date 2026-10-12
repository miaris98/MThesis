"""Unit Tests for World on Rails (WoR) Architecture, Models, and Trainer."""
import pytest
import numpy as np
import torch

from src.models.world_on_rails import (
    WorldOnRailsPolicy,
    QwenWorldOnRailsPolicy,
    WorldModel,
    RailsDynamicProgramming,
    PIDController,
    load_wor_model
)
from src.training.wor_dataset import WorldOnRailsDataset, create_wor_dataloader
from src.training.wor_trainer import WorldOnRailsTrainer
from src.training.wor_eval import waypoint_losses
from src.agents.wor_agent import WorldOnRailsAgent


def test_wor_policy_forward_and_act():
    """Tests WorldOnRailsPolicy forward pass and act() method."""
    policy = WorldOnRailsPolicy(
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False
    )
    policy.eval()

    B = 2
    dummy_rgb = torch.randn(B, 3, 256, 256)
    dummy_speed = torch.tensor([[10.0], [20.0]])
    dummy_cmd = torch.tensor([1, 2])

    out = policy(dummy_rgb, dummy_speed, dummy_cmd)

    assert "q_map" in out
    assert "rail_q" in out
    assert "waypoints" in out
    assert "selected_waypoints" in out

    assert out["q_map"].shape == (B, 6, 16, 16)
    assert out["rail_q"].shape == (B, 6, 9)
    assert out["selected_waypoints"].shape == (B, 5, 2)

    # Test act() method
    steer, throttle, brake = policy.act(
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        speed=15.0,
        command=2,
        device="cpu"
    )
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_qwen_wor_policy_forward_and_act():
    """Tests QwenWorldOnRailsPolicy (Qwen transformer decision head, offline WoR
    variant) forward pass, act(), and that its trunk lands in the ~100M param range,
    same as the PPO-side QwenDecisionTransformer at the same model_size."""
    policy = QwenWorldOnRailsPolicy(
        backbone_name="resnet18",
        pretrained=False,
        freeze_backbone=False,
        model_size="100m"
    )
    policy.eval()

    trunk_params = sum(p.numel() for p in policy.trunk.parameters())
    assert 90_000_000 < trunk_params < 120_000_000

    B = 2
    dummy_rgb = torch.randn(B, 3, 256, 256)
    dummy_speed = torch.tensor([[10.0], [20.0]])
    dummy_cmd = torch.tensor([1, 2])
    dummy_route = torch.randn(B, 4, 2)

    out = policy(dummy_rgb, dummy_speed, dummy_cmd, dummy_route)

    assert "rail_q" in out
    assert "waypoints" in out
    assert "selected_waypoints" in out
    assert out["rail_q"].shape == (B, 6, 9)
    assert out["waypoints"].shape == (B, 6, 5, 2)
    assert out["selected_waypoints"].shape == (B, 5, 2)

    steer, throttle, brake = policy.act(
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        speed=15.0,
        command=2,
        device="cpu",
        route=np.zeros((4, 2), dtype=np.float32)
    )
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_qwen_wor_dataset_and_trainer(tmp_path):
    """Confirms QwenWorldOnRailsPolicy is a drop-in for WorldOnRailsTrainer - same
    forward(rgb, speed, command, route) -> dict contract as the CNN-headed policy,
    so the offline distillation pipeline trains it with zero trainer changes."""
    policy = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, model_size="100m")
    trainer = WorldOnRailsTrainer(
        model=policy,
        data_dir=str(tmp_path / "fake_data"),
        save_dir=str(tmp_path / "checkpoints"),
        batch_size=4,
        num_workers=0,
        device="cpu",
        synthetic_samples=8
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "total_loss" in metrics
    assert "wp_loss" in metrics
    assert metrics["total_loss"] > 0


def test_carla_pretrained_backbone_loading(tmp_path):
    """Tests loading CARLA-domain pretrained perception weights (e.g. LAV / TransFuser++)."""
    # 1. Create a dummy CARLA checkpoint with perception. prefix
    dummy_backbone = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False)
    dummy_state = {"perception." + k: v for k, v in dummy_backbone.encoder.state_dict().items()}
    dummy_ckpt_path = str(tmp_path / "carla_lav_pretrained.pth")
    torch.save({"state_dict": dummy_state}, dummy_ckpt_path)

    # 2. Instantiate policy with CARLA pretrained weights
    policy_carla = WorldOnRailsPolicy(
        backbone_name="resnet18",
        pretrained=False,
        weights_path=dummy_ckpt_path
    )
    policy_carla.eval()

    dummy_rgb = torch.randn(1, 3, 256, 256)
    feats = policy_carla.encoder(dummy_rgb)
    assert feats.shape == (1, 512, 8, 8)


def test_rails_dynamic_programming():
    """Tests backward dynamic programming Bellman iteration."""
    dp = RailsDynamicProgramming(discount=0.9, num_rails=9, horizon=5)
    
    T, K = 10, 9
    rewards = np.ones((T, K), dtype=np.float32)
    collisions = np.zeros((T, K), dtype=bool)
    collisions[5, 2] = True  # Collision at step 5 on rail 2

    q_vals = dp.solve_trajectory_q_values(rewards, collision_mask=collisions)

    assert q_vals.shape == (T, K)
    assert q_vals[5, 2] < q_vals[5, 0]  # Collision rail has lower Q value


def test_world_model():
    """Tests learned WorldModel transition network."""
    wm = WorldModel(state_dim=64, num_rails=9, feature_dim=128)
    wm.eval()

    vis_feats = torch.randn(2, 128)
    ego_state = torch.randn(2, 64)
    action = torch.randn(2, 9)

    next_state, collision_logits = wm(vis_feats, ego_state, action)
    assert next_state.shape == (2, 64)
    assert collision_logits.shape == (2, 9)


def test_wor_dataset_and_trainer(tmp_path):
    """Tests World on Rails dataset and a 1-epoch distillation training loop."""
    dataset = WorldOnRailsDataset(
        data_dir=str(tmp_path / "fake_data"),
        synthetic_samples=8
    )
    assert len(dataset) == 8
    sample = dataset[0]
    # uint8 HWC: the float32/NCHW conversion happens on the GPU in the trainer.
    assert sample["rgb"].shape == (256, 256, 3)
    assert sample["rgb"].dtype == torch.uint8
    # Ego-frame route: the policy's only working navigation input, since PDM-Lite
    # leaves the command enum at LANEFOLLOW on every frame.
    assert sample["route"].shape == (4, 2)
    assert sample["target_q"].shape == (9,)
    assert sample["target_waypoints"].shape == (5, 2)

    policy = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False)
    trainer = WorldOnRailsTrainer(
        model=policy,
        data_dir=str(tmp_path / "fake_data"),
        save_dir=str(tmp_path / "checkpoints"),
        batch_size=4,
        num_workers=0,
        device="cpu",
        synthetic_samples=8
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "total_loss" in metrics
    assert "q_loss" in metrics
    assert "wp_loss" in metrics
    assert metrics["total_loss"] > 0


def test_wor_agent():
    """Tests PCLA-compatible WorldOnRailsAgent initialization and step execution."""
    agent = WorldOnRailsAgent(
        checkpoint_path=None,
        backbone_name="resnet18",
        pretrained_backbone=False,
        device="cpu"
    )

    sensors = agent.sensors()
    assert len(sensors) >= 3

    dummy_input = {
        "rgb_front": (0, np.zeros((256, 256, 3), dtype=np.uint8)),
        "speed": (0, {"speed": 4.0}),
        "command": 2
    }
    control = agent.run_step(dummy_input)
    assert hasattr(control, "steer") or (isinstance(control, dict) and "steer" in control)
    agent.destroy()


def test_wor_reward_function():
    """Tests World on Rails paper reward function calculations."""
    from src.envs.rewards import make_reward, WorldOnRailsReward
    reward_fn = make_reward("wor")
    assert isinstance(reward_fn, WorldOnRailsReward)

    # 1. Test normal forward progress
    state_normal = {
        "speed_kmh": 20.0,
        "heading_cos": 1.0,
        "lateral_dist": 0.1,
        "is_collision": False,
        "is_off_road": False,
        "is_at_red_light": False
    }
    r_step, info = reward_fn.compute_reward(state_normal, dt=0.05)
    assert r_step > 0.0
    assert info["r_progress"] > 0.0
    assert info["r_terminal"] == 0.0

    # 2. Test collision penalty
    state_collision = {
        "speed_kmh": 10.0,
        "heading_cos": 1.0,
        "lateral_dist": 0.0,
        "is_collision": True,
        "is_off_road": False,
        "is_at_red_light": False
    }
    r_coll, info_coll = reward_fn.compute_reward(state_collision)
    assert r_coll <= -20.0
    assert info_coll["r_terminal"] == -25.0


# --------------------------------------------------------------------------- #
# Path-geometry loss terms (src/training/wor_eval.py)
# --------------------------------------------------------------------------- #

def _wp_batch(pred, target):
    """Wraps two (5, 2) waypoint paths into the dict waypoint_losses expects."""
    pred_t = torch.tensor(pred, dtype=torch.float32).unsqueeze(0)
    target_t = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    out = {"selected_waypoints": pred_t, "selected_rail_q": torch.zeros(1, 9)}
    return out, target_t, torch.zeros(1, 9)


#: Expert drives straight ahead at 2 m per waypoint.
_STRAIGHT = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]]


def test_geometry_terms_default_to_the_original_objective():
    """The four-town comparison, the three seeds and the size curve are all stated in
    the loss as it was *before* the heading/curvature terms existed. If the defaults
    changed that number, every one of those results would silently stop being
    comparable, so pin it: unweighted, the total must still be exactly
    `longitudinal + 3 * lateral`."""
    curving = [[2.0, 0.3], [4.0, 1.2], [6.0, 2.7], [8.0, 4.8], [10.0, 7.5]]
    out, target, target_q = _wp_batch(curving, _STRAIGHT)

    losses = waypoint_losses(out, target, target_q)

    pred_t, tgt_t = out["selected_waypoints"], target
    expected = (torch.nn.functional.l1_loss(pred_t[..., 0], tgt_t[..., 0])
                + 3.0 * torch.nn.functional.l1_loss(pred_t[..., 1], tgt_t[..., 1]))
    assert torch.allclose(losses["total"], expected, atol=1e-6)
    assert torch.allclose(losses["total"], losses["wp"], atol=1e-6)


def test_geometry_terms_are_zero_on_a_perfect_path():
    out, target, target_q = _wp_batch(_STRAIGHT, _STRAIGHT)
    losses = waypoint_losses(out, target, target_q,
                             heading_loss_weight=1.0, curvature_loss_weight=1.0)
    assert losses["heading"].item() < 1e-6
    assert losses["curvature"].item() < 1e-6


def test_curvature_separates_paths_a_waypoint_l1_cannot():
    """A steady lateral offset and a zig-zag across the same offset carry identical
    per-waypoint L1. Only the second one makes the PID's derivative term chatter, which
    is exactly the failure the position loss is blind to."""
    offset = [[2.0, 0.4], [4.0, 0.4], [6.0, 0.4], [8.0, 0.4], [10.0, 0.4]]
    zigzag = [[2.0, 0.4], [4.0, -0.4], [6.0, 0.4], [8.0, -0.4], [10.0, 0.4]]

    smooth = waypoint_losses(*_wp_batch(offset, _STRAIGHT))
    jagged = waypoint_losses(*_wp_batch(zigzag, _STRAIGHT))

    assert torch.allclose(smooth["wp"], jagged["wp"], atol=1e-6)
    assert jagged["curvature"].item() > 10 * smooth["curvature"].item()
    assert jagged["heading"].item() > 10 * smooth["heading"].item()


def test_heading_is_invariant_to_path_length():
    """The whole point of the term: it cannot be swamped by longitudinal magnitude the
    way the position L1 is (73% of which is forward displacement the controller's
    steering never reads). Scaling a path threefold triples the L1 and must leave the
    heading error untouched."""
    short = [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5]]
    long = [[3.0 * x, 3.0 * y] for x, y in short]
    short_target = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    long_target = [[3.0 * x, 0.0] for x, _ in short_target]

    near = waypoint_losses(*_wp_batch(short, short_target))
    far = waypoint_losses(*_wp_batch(long, long_target))

    assert far["wp"].item() == pytest.approx(3.0 * near["wp"].item(), rel=1e-4)
    assert far["heading"].item() == pytest.approx(near["heading"].item(), abs=1e-5)


def test_heading_masks_a_stopped_vehicle_instead_of_producing_nan():
    """A stationary expert has zero-length segments, whose direction is undefined.
    Those must drop out rather than divide through a near-zero norm."""
    stopped = [[0.0, 0.0]] * 5
    out, target, target_q = _wp_batch([[0.02, 0.01]] * 5, stopped)

    losses = waypoint_losses(out, target, target_q,
                             heading_loss_weight=1.0, curvature_loss_weight=1.0)

    for key in ("heading", "curvature", "total"):
        assert torch.isfinite(losses[key]).all()
    assert losses["heading"].item() == 0.0


def test_geometry_terms_produce_finite_gradients():
    pred = torch.tensor(
        [[2.0, 0.3], [4.0, 1.2], [6.0, 2.7], [8.0, 4.8], [10.0, 7.5]]
    ).unsqueeze(0).requires_grad_(True)
    out = {"selected_waypoints": pred, "selected_rail_q": torch.zeros(1, 9)}

    losses = waypoint_losses(
        out, torch.tensor(_STRAIGHT).unsqueeze(0), torch.zeros(1, 9),
        wp_loss_weight=0.0, heading_loss_weight=1.0, curvature_loss_weight=1.0
    )
    losses["total"].backward()

    assert torch.isfinite(pred.grad).all()
    assert pred.grad.norm().item() > 0.0
