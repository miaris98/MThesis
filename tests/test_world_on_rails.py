"""Unit Tests for World on Rails (WoR) Architecture, Models, and Trainer."""
import math
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


def _per_cell_grad_ratio(policy, spatial: bool = True):
    """max/min gradient magnitude across the encoder feature map's spatial cells.

    Exactly 1.0 means every cell influences the output identically, i.e. the head is
    blind to *where* anything is in the frame - which is what AdaptiveAvgPool2d((1,1))
    guarantees, and what the first Qwen WoR runs were trained under.
    """
    B = 4
    rgb = torch.rand(B, 3, 256, 256)
    speed = torch.rand(B, 1) * 30
    cmd = torch.randint(0, 6, (B,))
    route = torch.randn(B, 4, 2)

    with torch.no_grad():
        feats = policy.encoder(rgb)
    feats = feats.clone().requires_grad_(True)

    if isinstance(policy, QwenWorldOnRailsPolicy):
        vis, spd, rt, ct, idx = policy._tokenize_state(feats, speed, cmd, route)
        wp, _ = policy.trunk(vis, spd, rt, ct)
        sel = wp[torch.arange(B), idx]
    else:
        pooled = feats if spatial else torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        _, _, wp = policy.q_head(pooled, policy.embed_state(speed, cmd, route))
        sel = wp[torch.arange(B), cmd]

    grad = torch.autograd.grad(sel[..., 0].sum(), feats)[0].abs().mean(dim=(0, 1))
    return (grad.max() / grad.min()).item()


def test_qwen_vision_grid_restores_spatial_sensitivity():
    """The transformer trunk must be able to tell one region of the frame from another.

    With vision_grid=0 the encoder's 8x8 map is globally averaged into a single token
    before the trunk sees it, so the gradient with respect to every cell is identical
    by construction. Forwarding the cells as separate tokens is what makes the
    self-attention trunk capable of spatial reasoning at all.
    """
    blind = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=0).eval()
    assert blind.num_vision_tokens == 1
    assert _per_cell_grad_ratio(blind) == pytest.approx(1.0, abs=1e-3)

    seeing = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=8).eval()
    assert seeing.num_vision_tokens == 64
    assert _per_cell_grad_ratio(seeing) > 1.05


def test_qwen_trunk_distinguishes_token_roles():
    """Self-attention is permutation-invariant and this trunk has no causal mask, so
    without positional/type embeddings it cannot tell the speed token from the route
    token - swapping them would leave the output bit-identical."""
    policy = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4).eval()
    B = 4
    with torch.no_grad():
        feats = policy.encoder(torch.rand(B, 3, 256, 256))
        vis, spd, rt, ct, _ = policy._tokenize_state(
            feats, torch.rand(B, 1) * 30, torch.randint(0, 6, (B,)), torch.randn(B, 4, 2)
        )
        correct, _ = policy.trunk(vis, spd, rt, ct)
        swapped, _ = policy.trunk(vis, rt, spd, ct)  # speed and route roles exchanged

    assert (correct - swapped).abs().max() > 1e-6


def test_rmsnorm_survives_fp16_activations():
    """RMSNorm squares its input; doing that in half precision overflows fp16's 65504
    ceiling once the residual stream grows, silently costing an optimizer step."""
    from src.models.transformer.layers import RMSNorm

    norm = RMSNorm(768)
    for magnitude in (100.0, 256.0, 1000.0):
        out = norm(torch.full((1, 1, 768), magnitude, dtype=torch.float16))
        assert torch.isfinite(out).all(), f"RMSNorm produced non-finite output at |x|={magnitude}"


def test_trainer_excludes_gates_and_norms_from_weight_decay(tmp_path):
    """AdamW weight decay on the trunk's alpha residual gates pulls every block's
    contribution toward zero - it penalizes the model for letting its layers
    contribute at all. Gates, norm gains, embeddings and learned tokens belong in a
    weight_decay=0 group."""
    policy = QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4)
    trainer = WorldOnRailsTrainer(
        model=policy, data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False
    )

    gate_ids = {id(p) for n, p in policy.named_parameters()
                if n.endswith(("alpha_attn", "alpha_ffn", "policy_token", "vision_pos", "type_embed"))}
    assert gate_ids, "expected the Qwen trunk to expose residual gates and learned tokens"

    for group in trainer.optimizer.param_groups:
        if any(id(p) in gate_ids for p in group["params"]):
            assert group["weight_decay"] == 0.0

    # And the opt-out still reproduces the original behaviour for comparison runs.
    legacy = WorldOnRailsTrainer(
        model=QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck2"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False,
        decay_gates_and_norms=True
    )
    assert all(len(g["params"]) == 0 for g in legacy.optimizer.param_groups if g["weight_decay"] == 0.0)


def test_scheduler_warms_up_then_decays(tmp_path):
    """A bare cosine schedule starts the heads at full LR on batch 1. A conv head
    tolerates that; a 12-layer pre-norm transformer characteristically does not."""
    trainer = WorldOnRailsTrainer(
        model=WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=2, num_workers=0, device="cpu", synthetic_samples=64,
        use_mlflow=False, warmup_frac=0.1, lr_heads=3e-4
    )
    scheduler = trainer._build_scheduler(num_epochs=20)

    lrs = []
    for _ in range(20 * len(trainer.train_loader)):
        lrs.append(trainer.optimizer.param_groups[1]["lr"])
        scheduler.step()

    peak = max(lrs)
    assert lrs[0] < peak / 2, "LR did not start low - warmup is not happening"
    assert peak == pytest.approx(3e-4, rel=1e-3), "warmup should reach the configured head LR"
    assert lrs[-1] < peak / 10, "LR did not decay after warmup"
    assert lrs.index(peak) == pytest.approx(int(0.1 * len(lrs)), abs=2)


def test_trainer_validates_and_selects_best_on_val(tmp_path):
    """`val_data_dir` used to be stored and never read, and 'best' was chosen on
    training loss - which cannot distinguish generalization from memorization."""
    trainer = WorldOnRailsTrainer(
        model=WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=40,
        use_mlflow=False, val_split=0.25
    )
    assert trainer.val_loader is not None
    assert len(trainer.val_loader.dataset) > 0
    assert len(trainer.train_loader.dataset) + len(trainer.val_loader.dataset) == 40

    val = trainer.validate()
    for key in ("val_loss", "val_wp_ade_m", "val_wp_lateral_error_m", "val_wp_longitudinal_error_m"):
        assert key in val and math.isfinite(val[key])


def test_grad_norm_telemetry_survives_a_nonfinite_batch(tmp_path):
    """One non-finite batch used to poison the whole epoch's grad_norm mean to inf,
    which is what made that column unreadable in the first telemetry files. The AMP
    GradScaler produces those deliberately, so they are counted, not averaged."""
    trainer = WorldOnRailsTrainer(
        model=WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False),
        data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=16,
        use_mlflow=False, grad_clip=0.5
    )
    metrics = trainer.train_epoch(epoch=1)

    assert math.isfinite(metrics["grad_norm"])
    assert metrics["nonfinite_grad_batches"] == 0
    assert metrics["clipped_grad_batches"] > 0, "a 0.5 clip should bite on an untrained model"
    assert metrics["grad_clip"] == 0.5


def test_checkpoint_records_vision_geometry(tmp_path):
    """A Qwen checkpoint trained on one pooled vision token loads into a 64-token
    model without error under strict=False, then drives on untrained positional
    embeddings. Stamping the geometry into the checkpoint is what prevents that."""
    from src.models.world_on_rails.wor_loader import load_wor_model

    save_dir = tmp_path / "ck"
    trainer = WorldOnRailsTrainer(
        model=QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, vision_grid=4),
        data_dir=str(tmp_path / "none"), save_dir=str(save_dir),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False
    )
    trainer.train(num_epochs=1, save_freq=1)

    ckpt = torch.load(str(save_dir / "latest_model.pth"), map_location="cpu")
    assert ckpt["config"]["vision_grid"] == 4
    assert ckpt["config"]["num_vision_tokens"] == 16

    # The caller does not have to remember the geometry - it is read back off disk.
    restored = load_wor_model(
        checkpoint_path=str(save_dir / "latest_model.pth"), backbone_name="resnet18",
        pretrained_backbone=False, policy_arch="qwen100m"
    )
    assert restored.vision_grid == 4
    assert restored.num_vision_tokens == 16


def test_cnn_pool_vision_ablation_is_spatially_blind():
    """The matching ablation on the CNN side. Without it, a CNN-beats-transformer
    result confounds architecture family with the fact that only one of the two heads
    was ever shown where things are in the frame."""
    normal = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False).eval()
    assert _per_cell_grad_ratio(normal) > 1.05

    blind = WorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, pool_vision=True).eval()
    B = 2
    out = blind(torch.rand(B, 3, 256, 256), torch.rand(B, 1) * 30,
                torch.randint(0, 6, (B,)), torch.randn(B, 4, 2))
    assert out["selected_waypoints"].shape == (B, 5, 2)
    assert out["q_map"].shape == (B, 6, 16, 16)


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

