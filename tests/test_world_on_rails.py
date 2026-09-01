"""Unit Tests for World on Rails (WoR) Architecture, Models, and Trainer."""
import pytest
import numpy as np
import torch

from src.models.world_on_rails import (
    WorldOnRailsPolicy,
    WorldModel,
    RailsDynamicProgramming,
    PIDController,
    load_wor_model
)
from src.training.wor_dataset import WorldOnRailsDataset, create_wor_dataloader
from src.training.wor_trainer import WorldOnRailsTrainer
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
    assert sample["rgb"].shape == (3, 256, 256)
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
    assert "steer" in control or hasattr(control, "steer")
    agent.destroy()
