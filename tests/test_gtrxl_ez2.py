"""Unit tests for IMPALA-CNN + GTrXL + EfficientZero v2 Policy Architecture."""
import pytest
import torch
import torch.nn as nn

from atari_qwen.models.gtrxl_layers import GRUGating, GTrXLBlock
from atari_qwen.models.efficientzero_v2_predictor import EfficientZeroV2Predictor
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent


def test_gru_gating_identity_init():
    dim = 64
    gate = GRUGating(dim, bg_init=2.0)
    x = torch.randn(2, 10, dim)
    y = torch.randn(2, 10, dim)
    
    out = gate(x, y)
    assert out.shape == x.shape
    # With bg_init=2.0, z ~ sigmoid(-2) = 0.119, so out is heavily dominated by x
    diff = (out - x).abs().mean().item()
    assert diff < 0.5, f"Initial gate output drifted too far from input x: {diff}"


def test_gtrxl_block_forward():
    dim = 128
    block = GTrXLBlock(dim=dim, num_heads=4, ffn_dim=256)
    x = torch.randn(2, 16, dim)
    out = block(x)
    assert out.shape == (2, 16, dim)
    assert torch.isfinite(out).all()


def test_efficientzero_v2_predictor_unroll():
    latent_dim = 128
    action_dim = 4
    K = 5
    predictor = EfficientZeroV2Predictor(latent_dim=latent_dim, action_dim=action_dim, unroll_steps=K)
    
    z_0 = torch.randn(8, latent_dim)
    actions = torch.randint(0, action_dim, (8, K))
    
    res = predictor.unroll_trajectory(z_0, actions)
    assert len(res["latent_states"]) == K
    assert len(res["rewards"]) == K
    assert len(res["values"]) == K
    assert len(res["projections"]) == K
    
    assert res["latent_states"][0].shape == (8, latent_dim)
    assert res["rewards"][0].shape == (8,)
    assert res["values"][0].shape == (8,)


def test_impala_gtrxl_agent_end_to_end():
    B = 2
    obs = torch.randint(0, 255, (B, 4, 84, 84), dtype=torch.uint8)
    agent = ImpalaGTrXLAgent(
        action_dim=4,
        in_channels=4,
        embed_dim=128,
        depth=2,
        num_heads=4,
        ffn_dim=256,
        unroll_steps=3
    )
    
    # 1. Forward pass
    logits, value, latent_z = agent(obs)
    assert logits.shape == (B, 4)
    assert value.shape == (B, 1)
    assert latent_z.shape == (B, 128)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    
    # 2. Branch unroll
    actions = torch.randint(0, 4, (B, 3))
    rollout = agent.unroll_branches(latent_z, actions)
    assert len(rollout["latent_states"]) == 3
    
    # 3. Backprop test
    target_val = torch.ones(B, 1)
    loss = (value - target_val).pow(2).mean() + logits.sum() + rollout["rewards"][0].sum()
    loss.backward()
    
    # Verify gradients reach visual encoder and gate parameters
    assert agent.visual_encoder.stage1.conv.weight.grad is not None
    assert agent.blocks[0].gate1.bg.grad is not None
    assert agent.predictor.dynamics.action_embed.weight.grad is not None
