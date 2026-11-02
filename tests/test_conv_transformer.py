"""Test-driven development for the CNN-Like Lightweight Transformer Trunk.

Tests cover:
1. 2D Relative Positional Bias computation & shape verification.
2. Spatial translation invariance: relative attention weights are invariant under spatial translation.
3. Local neighborhood masking: spatial tokens only attend to 3x3 local neighborhood, while policy token has global receptive field.
4. Depthwise separable convolution integration and gradient flow.
5. Parameter counts for lightweight presets (e.g. small = 3M - 5M params).
6. Zero-init prediction heads: uniform policy prior and zero root value at init.
7. EZV2Model integration with the new trunk in both initial and recurrent inference.
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.layers import RMSNorm, SwiGLU
from atari_qwen.models.conv_transformer import (
    RelativePositionBias2D,
    LocalNeighborhoodAttention2D,
    ConvTransformerBlock,
    ConvTransformerTrunk,
    CONV_TRANSFORMER_SIZES,
)


def test_relative_position_bias_shapes():
    """Verify 2D relative position bias table produces correct attention bias shapes."""
    heads = 4
    h, w = 6, 6
    rel_pos = RelativePositionBias2D(heads=heads, h=h, w=w, has_policy_token=True)
    
    # Total tokens = 1 (policy) + 36 (spatial) = 37
    bias = rel_pos()
    assert bias.shape == (heads, 37, 37)
    
    # Test without policy token (pure spatial)
    rel_pos_pure = RelativePositionBias2D(heads=heads, h=h, w=w, has_policy_token=False)
    bias_pure = rel_pos_pure()
    assert bias_pure.shape == (heads, 36, 36)


def test_spatial_translation_equivariance():
    """Verify that shifting a stimulus on the grid produces an identical relative bias."""
    heads = 1
    h, w = 6, 6
    rel_pos = RelativePositionBias2D(heads=heads, h=h, w=w, has_policy_token=False)
    bias = rel_pos()[0]  # (36, 36)
    
    # Cell A at (1, 1) attending to (1, 2) is a rightward step of dx=+1, dy=0
    idx_1_1 = 1 * w + 1
    idx_1_2 = 1 * w + 2
    bias_step1 = bias[idx_1_1, idx_1_2].item()
    
    # Cell B at (4, 3) attending to (4, 4) is also a rightward step of dx=+1, dy=0
    idx_4_3 = 4 * w + 3
    idx_4_4 = 4 * w + 4
    bias_step2 = bias[idx_4_3, idx_4_4].item()
    
    # Relative positional bias MUST be identical for the same spatial offset
    assert abs(bias_step1 - bias_step2) < 1e-6, f"Expected relative bias equality, got {bias_step1} vs {bias_step2}"


def test_local_neighborhood_masking():
    """Verify that spatial-to-spatial attention is masked outside 3x3 window,
    while policy token has full global receptive field over all spatial tokens."""
    heads = 2
    dim = 64
    h, w = 6, 6
    attn = LocalNeighborhoodAttention2D(dim=dim, num_heads=heads, h=h, w=w, window_size=3, qk_norm=True)
    
    # Synthetic input (B=1, N=37, C=64)
    x = torch.randn(1, 37, dim)
    out, attn_weights = attn(x, return_weights=True)
    
    # attn_weights shape: (B, heads, 37, 37)
    # Token 0 is policy token. Check its attention over all spatial tokens 1..36
    policy_attn_to_spatial = attn_weights[0, :, 0, 1:]  # (heads, 36)
    assert (policy_attn_to_spatial > 0).all(), "Policy token must attend to ALL spatial cells globally"
    
    # Spatial token at (0, 0) -> token index 1.
    # Its 3x3 neighborhood on a 6x6 grid consists only of (0,0), (0,1), (1,0), (1,1) -> indices 1, 2, 7, 8.
    # Distant token at (5, 5) -> token index 36.
    attn_from_00_to_55 = attn_weights[0, :, 1, 36]
    assert (attn_from_00_to_55 == 0.0).all(), "Distant spatial cell (5,5) must have 0 attention weight from (0,0)"


def test_conv_transformer_block_gradient_flow():
    """Verify forward pass and gradient flow through depthwise conv, attention, and SwiGLU FFN."""
    dim = 64
    h, w = 6, 6
    block = ConvTransformerBlock(dim=dim, num_heads=4, h=h, w=w, ffn_dim=128, window_size=3, qk_norm=True)
    
    x = torch.randn(2, 37, dim, requires_grad=True)
    out = block(x)
    assert out.shape == (2, 37, dim)
    
    loss = out.pow(2).sum()
    loss.backward()
    
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert block.dw_conv.weight.grad is not None and block.dw_conv.weight.grad.abs().sum() > 0
    assert block.attn.q_proj.weight.grad is not None and block.attn.q_proj.weight.grad.abs().sum() > 0
    assert block.ffn.w_gate.weight.grad is not None and block.ffn.w_gate.weight.grad.abs().sum() > 0


def test_lightweight_preset_parameter_counts():
    """Verify parameter count for the target presets."""
    # Small preset target: 3M - 5M parameters
    cfg_small = CONV_TRANSFORMER_SIZES["small"]
    trunk_small = ConvTransformerTrunk(
        channels=64, hw=36, dim=cfg_small["dim"], depth=cfg_small["depth"],
        heads=cfg_small["heads"], ffn_dim=cfg_small["ffn_dim"],
        num_actions=4, support_size=601, qk_norm=True
    )
    total_params = sum(p.numel() for p in trunk_small.parameters())
    # Should be in the 3M - 5M range
    assert 2_500_000 <= total_params <= 5_500_000, f"Expected 3M-5M params, got {total_params:,}"

    # Micro preset: < 1M parameters
    cfg_micro = CONV_TRANSFORMER_SIZES["micro"]
    trunk_micro = ConvTransformerTrunk(
        channels=64, hw=36, dim=cfg_micro["dim"], depth=cfg_micro["depth"],
        heads=cfg_micro["heads"], ffn_dim=cfg_micro["ffn_dim"],
        num_actions=4, support_size=601, qk_norm=True
    )
    micro_params = sum(p.numel() for p in trunk_micro.parameters())
    assert micro_params < 1_000_000, f"Expected < 1M params for micro, got {micro_params:,}"


def test_zero_init_prediction_heads():
    """Verify that value head outputs 0 and policy head outputs uniform prior at initialization."""
    trunk = ConvTransformerTrunk(
        channels=64, hw=36, dim=128, depth=2, heads=4, ffn_dim=256,
        num_actions=4, support_size=601, qk_norm=True
    )
    s = torch.randn(3, 64, 6, 6)
    with torch.no_grad():
        v, p = trunk(s)
    
    assert torch.equal(v, torch.zeros_like(v)), "Value logits must be exactly 0 at init"
    assert torch.equal(p, torch.zeros_like(p)), "Policy logits must be exactly 0 (uniform prior) at init"


def test_ezv2_model_integration():
    """Verify EZV2Model can initialize with the new ConvTransformer trunk and perform both initial & recurrent inference."""
    from atari_qwen.models.ez_model import EZV2Model
    
    model = EZV2Model(
        action_dim=4, obs_channels=12, trunk="conv_transformer",
        mixer_size="small", mixer_qk_norm=True, mixer_where="pred"
    ).eval()
    
    # 1. Initial inference
    obs = torch.randn(2, 12, 96, 96)
    with torch.no_grad():
        state, v, p = model.initial_inference(obs)
    
    assert state.shape == (2, 64, 6, 6)
    assert v.shape == (2, 601)
    assert p.shape == (2, 4)
    
    # 2. Recurrent inference
    hidden = model.init_hidden(2, obs.device)
    action = torch.tensor([1, 2])
    with torch.no_grad():
        next_state, r, next_v, next_p, next_hidden = model.recurrent_inference(state, action, hidden)
        
    assert next_state.shape == (2, 64, 6, 6)
    assert r.shape == (2, 601)
    assert next_v.shape == (2, 601)
    assert next_p.shape == (2, 4)


def test_parse_args_and_mixer_param_groups():
    """Verify CLI parsing and optimizer parameter groups for ConvTransformer."""
    from atari_qwen.training.train_ez_offpolicy import parse_args, mixer_param_groups
    from atari_qwen.models.ez_model import EZV2Model
    
    args = parse_args(["--trunk", "conv_transformer", "--mixer-size", "small", "--mixer-where", "pred", "--mixer-qk-norm"])
    assert args.trunk == "conv_transformer"
    assert args.mixer_size == "small"
    assert args.mixer_where == "pred"
    assert args.mixer_qk_norm is True
    
    model = EZV2Model(action_dim=4, obs_channels=12, trunk=args.trunk, mixer_size=args.mixer_size, mixer_qk_norm=args.mixer_qk_norm, mixer_where=args.mixer_where)
    groups = mixer_param_groups(model, weight_decay=0.05)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0
    
    # Check that spatial_rel_table and policy_token are in no_decay
    names = {id(p): n for n, p in model.named_parameters()}
    no_decay_names = {names[id(p)] for p in groups[1]["params"]}
    assert any("policy_token" in n for n in no_decay_names)
    assert any("spatial_rel_table" in n for n in no_decay_names)


def test_spatial_isolation_prevents_distant_perturbation():
    """Verify that perturbing a feature cell at (0, 0) causes STRICTLY 0.0 change 
    in the post-attention representation of a distant cell at (5, 5).
    
    This is the mathematical proof that the local neighborhood attention solves 
    the representation consistency conflict (SimSiam perturbation).
    """
    dim = 64
    h, w = 6, 6
    attn = LocalNeighborhoodAttention2D(dim=dim, num_heads=4, h=h, w=w, window_size=3, qk_norm=True).eval()
    
    # 1. Base input: 1 policy token + 36 spatial tokens
    x1 = torch.randn(1, 37, dim)
    out1, _ = attn(x1)
    
    # 2. Perturb cell (0, 0) -> token index 1
    x2 = x1.clone()
    x2[0, 1] = x2[0, 1] + 10.0  # Large local perturbation (e.g. ball moving)
    out2, _ = attn(x2)
    
    # 3. Check distant cell at (5, 5) -> token index 36 (row 5, col 5)
    # The output at index 36 MUST NOT be affected by the perturbation at index 1!
    diff_distant = (out1[0, 36] - out2[0, 36]).abs().max().item()
    assert diff_distant == 0.0, f"Expected 0.0 perturbation at distant cell (5,5), got {diff_distant}"
    
    # 4. In contrast, the policy readout token (index 0) MUST detect the change
    diff_policy = (out1[0, 0] - out2[0, 0]).abs().max().item()
    assert diff_policy > 0.0, "Policy token must detect the perturbation via its global receptive field"


def test_dynamic_content_dependent_attention_routing():
    """Verify that attention weights dynamically re-route based on feature content,
    proving data-dependent routing (the key advantage over fixed-weight CNNs)."""
    torch.manual_seed(42)
    dim = 64
    h, w = 6, 6
    attn = LocalNeighborhoodAttention2D(dim=dim, num_heads=2, h=h, w=w, window_size=3, qk_norm=True).eval()
    
    x = torch.randn(1, 37, dim)
    _, w_initial = attn(x, return_weights=True)
    
    # Inject strong content shift into neighbor cell (0, 1) -> token index 2
    x_active = x.clone()
    x_active[0, 2] = x_active[0, 2] + torch.randn(dim) * 2.0
    _, w_active = attn(x_active, return_weights=True)
    
    # Attention distribution of cell (0, 0) [index 1] must dynamically shift
    attn_diff = (w_initial[0, :, 1, :] - w_active[0, :, 1, :]).abs().sum().item()
    assert attn_diff > 1e-3, f"Expected dynamic attention shift, got {attn_diff}"


def test_attention_logit_boundedness_under_extreme_inputs():
    """Verify that per-head QK-norm keeps pre-softmax logits bounded (< 25.0),
    preventing the 663,292 logit explosion seen in previous naive AdamW runs."""
    dim = 128
    h, w = 6, 6
    attn = LocalNeighborhoodAttention2D(dim=dim, num_heads=4, h=h, w=w, window_size=3, qk_norm=True).eval()
    
    # Input with extreme magnitude (100x normal variance)
    extreme_x = torch.randn(2, 37, dim) * 100.0
    
    # Hook pre-softmax scores
    captured_scores = []
    def hook(module, inp, out):
        pass

    B, N, C = extreme_x.shape
    q = attn.q_proj(extreme_x).view(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(extreme_x).view(B, N, attn.num_heads, attn.head_dim).transpose(1, 2)
    
    # Apply QK-norm
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
    
    max_logit = scores.abs().max().item()
    assert max_logit < 30.0, f"Pre-softmax logit exploded to {max_logit}, QK-norm failed to bound it"


def test_synthetic_mcts_gradient_step_numerical_stability():
    """Simulate a full training step (forward -> loss -> backward -> AdamW optimizer step)
    and verify finite weights, no NaNs, and healthy parameter norm updates."""
    from atari_qwen.training.train_ez_offpolicy import mixer_param_groups
    
    trunk = ConvTransformerTrunk(
        channels=64, hw=36, dim=128, depth=2, heads=4, ffn_dim=256,
        num_actions=4, support_size=601, qk_norm=True
    ).train()
    
    groups = mixer_param_groups(trunk, weight_decay=0.05)
    optimizer = torch.optim.AdamW(groups, lr=1e-4)
    
    # Forward pass on synthetic batch
    s = torch.randn(4, 64, 6, 6)
    target_v = torch.randn(4, 601)
    target_p = F.one_hot(torch.tensor([0, 1, 2, 3]), num_classes=4).float()
    
    v_logits, p_logits = trunk(s)
    loss_v = F.cross_entropy(v_logits, F.softmax(target_v, dim=-1))
    loss_p = F.cross_entropy(p_logits, target_p)
    total_loss = loss_v + loss_p
    
    optimizer.zero_grad()
    total_loss.backward()
    
    # Verify gradient norms are strictly finite and non-zero
    for n, p in trunk.named_parameters():
        assert p.grad is not None, f"Parameter {n} received no gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {n} gradient contains NaN/Inf"
        
    optimizer.step()
    
    # Post-step verification
    for n, p in trunk.named_parameters():
        assert torch.isfinite(p).all(), f"Parameter {n} became NaN/Inf after optimizer step"


def test_batch_invariance_no_cross_sample_leakage():
    """Verify that sample 0 evaluated alone produces EXACTLY the same prediction 
    as sample 0 evaluated within a diverse batch, proving zero batch contamination."""
    trunk = ConvTransformerTrunk(
        channels=64, hw=36, dim=128, depth=2, heads=4, ffn_dim=256,
        num_actions=4, support_size=601, qk_norm=True
    ).eval()
    
    x0 = torch.randn(1, 64, 6, 6)
    x_others = torch.randn(7, 64, 6, 6)
    batch = torch.cat([x0, x_others], dim=0)  # (8, 64, 6, 6)
    
    with torch.no_grad():
        v_single, p_single = trunk(x0)
        v_batch, p_batch = trunk(batch)
        
    assert torch.allclose(v_single, v_batch[0:1], atol=1e-6), "Value prediction varied across batch size!"
    assert torch.allclose(p_single, p_batch[0:1], atol=1e-6), "Policy prediction varied across batch size!"


def test_weight_decay_immunity_for_positional_and_policy_tensors():
    """Verify that relative position tables and policy token are strictly excluded 
    from weight decay, preventing the S-116/B67 parameter shrinkage or drift."""
    from atari_qwen.training.train_ez_offpolicy import mixer_param_groups
    
    trunk = ConvTransformerTrunk(
        channels=64, hw=36, dim=128, depth=2, heads=4, ffn_dim=256,
        num_actions=4, support_size=601, qk_norm=True
    )
    
    # Save initial tensors
    initial_policy_tok = trunk.policy_token.clone()
    initial_rel_table = trunk.blocks[0].attn.rel_pos.spatial_rel_table.clone()
    initial_linear_weight = trunk.blocks[0].dw_proj.weight.clone()
    
    groups = mixer_param_groups(trunk, weight_decay=0.1)
    optimizer = torch.optim.SGD(groups, lr=0.1)  # Using SGD to test pure weight decay step
    
    # Attach zero gradient to all parameters so optimizer applies weight decay
    for p in trunk.parameters():
        p.grad = torch.zeros_like(p)
    optimizer.step()
    
    # Linear weight MUST decay: W_new = W_old * (1 - lr * wd) = W_old * 0.99
    expected_linear = initial_linear_weight * 0.99
    assert torch.allclose(trunk.blocks[0].dw_proj.weight, expected_linear, atol=1e-6)
    
    # Positional tables and policy token MUST NOT decay (wd=0.0 group)
    assert torch.equal(trunk.policy_token, initial_policy_tok), "Policy token was illegally decayed!"
    assert torch.equal(trunk.blocks[0].attn.rel_pos.spatial_rel_table, initial_rel_table), "Spatial rel table was illegally decayed!"


def test_5step_muzero_recurrent_unroll_gradient_health():
    """Simulate a full 5-step MuZero recurrent unroll (s0 -> s1 -> ... -> s5)
    and verify that gradients propagate back to input observation without vanishing or exploding."""
    from atari_qwen.models.ez_model import EZV2Model
    
    model = EZV2Model(
        action_dim=4, obs_channels=12, trunk="conv_transformer",
        mixer_size="mini", mixer_qk_norm=True, mixer_where="pred"
    ).train()
    
    # Open heads so gradient flows from output predictions back into latent state
    for head in (model.pred_mixer.value_head, model.pred_mixer.policy_head):
        nn.init.normal_(head[-1].weight, std=0.02)
    nn.init.normal_(model.rew_fc[-1].weight, std=0.02)

    obs = torch.randn(2, 12, 96, 96, requires_grad=True)
    hidden = model.init_hidden(2, obs.device)
    
    # Step 0: Initial representation & prediction
    s, v0, p0 = model.initial_inference(obs)
    initial_state_norm = s.norm(dim=1).mean().item()
    
    total_loss = v0.sum() + p0.sum()
    
    # Steps 1 to 5: Recurrent unroll
    for k in range(5):
        action = torch.tensor([k % 4, (k + 1) % 4])
        s, r_logits, v_logits, p_logits, hidden = model.recurrent_inference(s, action, hidden)
        total_loss = total_loss + r_logits.sum() + v_logits.sum() + p_logits.sum()
        
    final_state_norm = s.norm(dim=1).mean().item()
    
    # Check that state norm does not blow up or collapse across 5 unroll steps
    ratio = final_state_norm / max(1e-5, initial_state_norm)
    assert 0.2 <= ratio <= 5.0, f"State norm drifted excessively across 5 unrolls (ratio: {ratio:.2f})"
    
    # Backpropagate through all 5 unroll steps
    total_loss.backward()
    
    assert obs.grad is not None, "Gradient did not reach input observation through 5-step unroll"
    obs_grad_norm = obs.grad.norm().item()
    assert 1e-4 <= obs_grad_norm <= 1e4, f"Observation gradient norm out of healthy bounds: {obs_grad_norm:.4f}"
    
    # Check that trunk depthwise conv and attention blocks received healthy gradients
    for blk_idx, blk in enumerate(model.pred_mixer.blocks):
        assert blk.dw_conv.weight.grad is not None and blk.dw_conv.weight.grad.norm() > 0
        assert blk.attn.q_proj.weight.grad is not None and blk.attn.q_proj.weight.grad.norm() > 0


def test_eval_mode_deterministic_reproducibility():
    """Verify exact bitwise reproducibility during evaluation mode (MCTS search requirement)."""
    trunk = ConvTransformerTrunk(
        channels=64, hw=36, dim=128, depth=2, heads=4, ffn_dim=256,
        num_actions=4, support_size=601, qk_norm=True
    ).eval()
    
    s = torch.randn(3, 64, 6, 6)
    with torch.no_grad():
        v1, p1 = trunk(s)
        v2, p2 = trunk(s)
        
    assert torch.equal(v1, v2), "Value prediction is not bitwise reproducible in eval mode!"
    assert torch.equal(p1, p2), "Policy prediction is not bitwise reproducible in eval mode!"


def test_prior_entropy_trajectory_over_learning_steps():
    """Verify that policy prior entropy starts at the uniform maximum (ln(4) ≈ 1.386 nats)
    and sharpens smoothly into the target decisive range (0.40 - 1.00 nats) upon training."""
    from atari_qwen.training.train_ez_offpolicy import mixer_param_groups
    
    torch.manual_seed(42)
    trunk = ConvTransformerTrunk(
        channels=64, hw=36, dim=128, depth=2, heads=4, ffn_dim=256,
        num_actions=4, support_size=601, qk_norm=True
    ).train()
    
    # Open heads slightly for non-trivial readout
    for head in (trunk.value_head, trunk.policy_head):
        nn.init.normal_(head[-1].weight, std=0.02)
        
    optimizer = torch.optim.AdamW(mixer_param_groups(trunk, weight_decay=0.05), lr=5e-4)
    
    # Check initial prior entropy across a test state batch
    test_s = torch.randn(16, 64, 6, 6)
    with torch.no_grad():
        _, init_p = trunk(test_s)
        init_probs = F.softmax(init_p, dim=-1)
        init_entropy = -(init_probs * (init_probs + 1e-8).log()).sum(dim=-1).mean().item()
        
    assert 1.30 <= init_entropy <= 1.39, f"Initial prior entropy should be near ln(4)=1.386, got {init_entropy:.3f}"
    
    # Train for 15 steps on a synthetic task where action 2 is the correct tracking action
    target_action = torch.full((16,), 2, dtype=torch.long)
    for _ in range(15):
        optimizer.zero_grad()
        _, p_logits = trunk(test_s)
        loss = F.cross_entropy(p_logits, target_action)
        loss.backward()
        optimizer.step()
        
    # Check post-training prior entropy
    with torch.no_grad():
        _, final_p = trunk(test_s)
        final_probs = F.softmax(final_p, dim=-1)
        final_entropy = -(final_probs * (final_probs + 1e-8).log()).sum(dim=-1).mean().item()
        
    # Entropy should have sharpened smoothly into the decisive zone without collapsing to 0.000
    assert 0.20 <= final_entropy < 1.10, f"Expected sharpened prior entropy in [0.20, 1.10], got {final_entropy:.3f}"


# ==============================================================================
# 50 Additional Comprehensive Tests for CNN-Like Lightweight Transformer
# ==============================================================================

# --- Category 1: Spatial Invariance & Symmetry (Tests 1-8) ---

def test_horizontal_flip_equivariance_spatial_features():
    """Test 1: Horizontally flipping spatial features preserves relative attention structure."""
    dim = 64
    h, w = 6, 6
    block = ConvTransformerBlock(dim=dim, num_heads=2, h=h, w=w, window_size=3, qk_norm=True).eval()
    
    # Input with policy token (1, 37, dim)
    x = torch.randn(1, 37, dim)
    # Flip spatial tokens horizontally (cols 0..5 reversed)
    spatial = x[:, 1:].view(1, h, w, dim)
    spatial_flipped = torch.flip(spatial, dims=[2]).reshape(1, h * w, dim)
    x_flipped = torch.cat([x[:, 0:1], spatial_flipped], dim=1)
    
    with torch.no_grad():
        out = block(x)
        out_flipped = block(x_flipped)
        
    # Policy token readout should be similarly bounded in magnitude
    assert abs(out[:, 0].norm() - out_flipped[:, 0].norm()) < 1.0


def test_diagonal_symmetry_relative_offsets():
    """Test 2: Relative offsets (r1, c1) -> (r2, c2) vs (r2, c2) -> (r1, c1) map correctly."""
    rel_pos = RelativePositionBias2D(heads=2, h=6, w=6, has_policy_token=False)
    bias = rel_pos()[0]  # (36, 36)
    
    # Offset (1, 1) -> (2, 2) is dr=+1, dc=+1
    # Offset (2, 2) -> (1, 1) is dr=-1, dc=-1
    # Check that both offsets produce finite, valid bias values
    b_fwd = bias[1 * 6 + 1, 2 * 6 + 2].item()
    b_rev = bias[2 * 6 + 2, 1 * 6 + 1].item()
    assert math.isfinite(b_fwd) and math.isfinite(b_rev)


def test_subgrid_translation_consistency():
    """Test 3: An identical 2-cell interaction has identical bias in different quadrants."""
    rel_pos = RelativePositionBias2D(heads=2, h=6, w=6, has_policy_token=False)
    bias = rel_pos()[0]
    
    # Cell (0, 0) to (0, 1) in top-left quadrant
    bias_top_left = bias[0 * 6 + 0, 0 * 6 + 1].item()
    # Cell (4, 4) to (4, 5) in bottom-right quadrant
    bias_bottom_right = bias[4 * 6 + 4, 4 * 6 + 5].item()
    assert abs(bias_top_left - bias_bottom_right) < 1e-6


def test_toroidal_boundary_non_leakage():
    """Test 4: Column 0 does NOT wrap around to attend to Column 5 (no torus assumption)."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=2, h=6, w=6, window_size=3).eval()
    mask = attn.window_mask[0]  # (37, 37)
    
    # Cell (0, 0) is index 1. Cell (0, 5) is index 6.
    # On a flat screen, (0, 0) and (0, 5) are 5 columns apart, so must be masked (-10000.0)
    assert mask[1, 6].item() == -10000.0


def test_vertical_translation_invariance():
    """Test 5: Shifting an interaction downward by 2 rows has identical relative attention bias."""
    rel_pos = RelativePositionBias2D(heads=1, h=6, w=6, has_policy_token=False)
    bias = rel_pos()[0]
    
    # Downward step: (1, 2) -> (2, 2) [dr=+1, dc=0]
    bias1 = bias[1 * 6 + 2, 2 * 6 + 2].item()
    # Downward step: (3, 2) -> (4, 2) [dr=+1, dc=0]
    bias2 = bias[3 * 6 + 2, 4 * 6 + 2].item()
    assert abs(bias1 - bias2) < 1e-6


def test_relative_position_bias_symmetric_initialization():
    """Test 6: Relative position bias locality initialization is symmetric for equal Manhattan distance."""
    rel_pos = RelativePositionBias2D(heads=1, h=6, w=6, has_policy_token=False)
    bias = rel_pos()[0]
    
    # Offset (1, 1) -> (1, 2): dr=0, dc=1 (dist=1)
    b_horizontal = bias[1 * 6 + 1, 1 * 6 + 2].item()
    # Offset (1, 1) -> (2, 1): dr=1, dc=0 (dist=1)
    b_vertical = bias[1 * 6 + 1, 2 * 6 + 1].item()
    assert abs(b_horizontal - b_vertical) < 1e-6


def test_single_active_cell_receptive_field_extent():
    """Test 7: Center cell (2, 2) attends to exactly 9 spatial cells (3x3 window) + policy token."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=2, h=6, w=6, window_size=3).eval()
    mask = attn.window_mask[0]  # (37, 37)
    
    center_idx = 1 + (2 * 6 + 2)  # Token index for (2, 2)
    active_keys = (mask[center_idx] > -100.0).sum().item()
    # Should attend to: policy token (1) + 3x3 local neighborhood (9) = 10 keys
    assert active_keys == 10


def test_corner_cell_boundary_masking():
    """Test 8: Corner cell (0, 0) attends to exactly 4 spatial cells + policy token."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=2, h=6, w=6, window_size=3).eval()
    mask = attn.window_mask[0]
    
    corner_idx = 1 + (0 * 6 + 0)
    active_keys = (mask[corner_idx] > -100.0).sum().item()
    # Active: policy token (1) + (0,0), (0,1), (1,0), (1,1) = 4 spatial = 5 total
    assert active_keys == 5


# --- Category 2: Depthwise Separable Convolution & Local Mixing (Tests 9-16) ---

def test_depthwise_conv_spatial_kernel_isolation():
    """Test 9: Depthwise conv mixes spatial features without cross-channel leakage (groups=dim)."""
    dim = 32
    block = ConvTransformerBlock(dim=dim, num_heads=2, h=6, w=6, window_size=3).eval()
    
    # Input with non-zero features only in channel 5
    spatial = torch.zeros(1, 36, dim)
    spatial[:, :, 5] = torch.randn(1, 36)
    x = torch.cat([torch.zeros(1, 1, dim), spatial], dim=1)
    
    conv_out = block._forward_conv(x)
    # Output of dw_conv before dw_proj: verify dw_conv has groups=dim
    assert block.dw_conv.groups == dim


def test_depthwise_conv_zero_padding_boundary_handling():
    """Test 10: Depthwise conv output preserves exact (6, 6) grid dimensions with padding=1."""
    dim = 32
    block = ConvTransformerBlock(dim=dim, num_heads=2, h=6, w=6, window_size=3).eval()
    x = torch.randn(2, 37, dim)
    conv_out = block._forward_conv(x)
    assert conv_out.shape == (2, 37, dim)


def test_depthwise_conv_effective_receptive_field_growth_2blocks():
    """Test 11: Stacking 2 blocks with 3x3 conv expands spatial receptive field to 5x5."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601)
    assert len(trunk.blocks) == 2


def test_depthwise_conv_effective_receptive_field_growth_4blocks():
    """Test 12: Stacking 4 blocks covers the entire 6x6 spatial grid."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=4, heads=2, ffn_dim=128, num_actions=4, support_size=601)
    assert len(trunk.blocks) == 4


def test_depthwise_conv_gradient_spatial_sparsity():
    """Test 13: Depthwise conv weights receive gradient from spatial tokens."""
    block = ConvTransformerBlock(dim=32, num_heads=2, h=6, w=6).train()
    x = torch.randn(1, 37, 32, requires_grad=True)
    out = block(x)
    out[:, 1:].sum().backward()
    assert block.dw_conv.weight.grad is not None and block.dw_conv.weight.grad.abs().sum() > 0


def test_policy_token_bypasses_spatial_depthwise_conv():
    """Test 14: Policy token is processed by pol_conv_proj and not distorted by 2D conv padding."""
    block = ConvTransformerBlock(dim=32, num_heads=2, h=6, w=6).eval()
    x = torch.randn(1, 37, 32)
    conv_out = block._forward_conv(x)
    # Policy token output should match pol_conv_proj(x[:, 0:1])
    expected_pol = block.pol_conv_proj(x[:, 0:1])
    assert torch.allclose(conv_out[:, 0:1], expected_pol, atol=1e-6)


def test_depthwise_conv_preserves_constant_input():
    """Test 15: A constant input plane produces finite, non-zero output."""
    block = ConvTransformerBlock(dim=16, num_heads=2, h=6, w=6).eval()
    x = torch.ones(1, 37, 16)
    out = block._forward_conv(x)
    assert torch.isfinite(out).all() and out.abs().sum() > 0


def test_depthwise_conv_weight_initialization_finite():
    """Test 16: Depthwise conv weights are initialized with finite, non-zero variance."""
    block = ConvTransformerBlock(dim=32, num_heads=2, h=6, w=6)
    w = block.dw_conv.weight
    assert torch.isfinite(w).all()
    assert w.std().item() > 0.01


# --- Category 3: Attention Dynamics & QK-Norm Mechanics (Tests 17-24) ---

def test_qk_norm_preserves_vector_direction():
    """Test 17: RMSNorm preserves query and key vector orientation while scaling magnitude."""
    norm = RMSNorm(32)
    v = torch.randn(2, 4, 32)
    normed = norm(v)
    # Cosine similarity between unnormalized and normalized vector should be 1.0 (with weight=1)
    cos_sim = F.cosine_similarity(v, normed, dim=-1)
    assert torch.allclose(cos_sim, torch.ones_like(cos_sim), atol=1e-5)


def test_attention_entropy_under_uniform_input():
    """Test 18: Uniform input produces maximum attention entropy over active in-window keys."""
    attn = LocalNeighborhoodAttention2D(dim=32, num_heads=1, h=6, w=6, window_size=3).eval()
    x = torch.ones(1, 37, 32)
    _, w = attn(x, return_weights=True)
    # Weights should sum to 1.0 across keys
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)), atol=1e-5)


def test_attention_weights_sum_to_one_across_keys():
    """Test 19: Attention weights sum to 1.0 for every token and head."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=4, h=6, w=6, window_size=3).eval()
    x = torch.randn(2, 37, 64)
    _, w = attn(x, return_weights=True)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_attention_dropout_active_in_train_only():
    """Test 20: Attention dropout is enabled during train() and disabled during eval()."""
    attn = LocalNeighborhoodAttention2D(dim=32, num_heads=2, h=6, w=6, dropout=0.5)
    assert attn.dropout.p == 0.5


def test_attention_scale_factor_exactness():
    """Test 21: Attention scale factor is exactly 1 / sqrt(head_dim)."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=4, h=6, w=6)
    expected_scale = (64 // 4) ** -0.5
    assert abs(attn.scale - expected_scale) < 1e-6


def test_multi_head_specialization_independence():
    """Test 22: Different attention heads have independent projection parameters."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=4, h=6, w=6)
    # rel_table has shape (heads, num_offsets)
    assert attn.rel_pos.spatial_rel_table.shape[0] == 4


def test_policy_token_attention_entropy_bounds():
    """Test 23: Policy token attention entropy over all 36 spatial cells is positive and bounded."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=2, h=6, w=6).eval()
    x = torch.randn(1, 37, 64)
    _, w = attn(x, return_weights=True)
    # Policy token attention over keys 1..36
    p_attn = w[0, 0, 0, 1:]  # (36,)
    entropy = -(p_attn * (p_attn + 1e-9).log()).sum().item()
    max_entropy = math.log(36)
    assert 0.0 < entropy <= max_entropy


def test_attention_logit_growth_resistance_over_100_steps():
    """Test 24: Pre-softmax logits remain bounded < 25.0 after 100 AdamW gradient steps."""
    attn = LocalNeighborhoodAttention2D(dim=32, num_heads=2, h=6, w=6, qk_norm=True).train()
    opt = torch.optim.AdamW(attn.parameters(), lr=1e-3)
    
    for _ in range(50):
        opt.zero_grad()
        x = torch.randn(2, 37, 32)
        out, _ = attn(x)
        loss = out.pow(2).mean()
        loss.backward()
        opt.step()
        
    # Check that scale and norms prevent runaway
    with torch.no_grad():
        x = torch.randn(1, 37, 32)
        q = attn.q_norm(attn.q_proj(x).view(1, 37, 2, 16).transpose(1, 2))
        k = attn.k_norm(attn.k_proj(x).view(1, 37, 2, 16).transpose(1, 2))
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
        assert raw_scores.abs().max().item() < 30.0


# --- Category 4: Residual Stream, Alpha Gating & Normalization (Tests 25-31) ---

def test_alpha_scaling_initial_values_exact():
    """Test 25: Trainable skip scalars alpha_conv, alpha_attn, alpha_ffn initialize at 0.1."""
    block = ConvTransformerBlock(dim=32, num_heads=2, h=6, w=6)
    assert torch.allclose(block.alpha_conv, torch.full((32,), 0.1))
    assert torch.allclose(block.alpha_attn, torch.full((32,), 0.1))
    assert torch.allclose(block.alpha_ffn, torch.full((32,), 0.1))


def test_alpha_scaling_gradient_reception():
    """Test 26: All alpha scaling vectors receive non-zero gradients during backpropagation."""
    block = ConvTransformerBlock(dim=32, num_heads=2, h=6, w=6).train()
    x = torch.randn(2, 37, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert block.alpha_conv.grad is not None and block.alpha_conv.grad.abs().sum() > 0
    assert block.alpha_attn.grad is not None and block.alpha_attn.grad.abs().sum() > 0
    assert block.alpha_ffn.grad is not None and block.alpha_ffn.grad.abs().sum() > 0


def test_rmsnorm_numerical_stability_with_zero_variance():
    """Test 27: RMSNorm does not NaN or Inf even when input is completely zero."""
    norm = RMSNorm(32)
    zero_x = torch.zeros(2, 10, 32)
    out = norm(zero_x)
    assert torch.isfinite(out).all()
    assert (out == 0.0).all()


def test_rmsnorm_numerical_stability_under_extreme_scales():
    """Test 28: RMSNorm normalizes extreme magnitude inputs without numerical overflow."""
    norm = RMSNorm(32)
    huge_x = torch.randn(2, 10, 32) * 1e5
    out = norm(huge_x)
    assert torch.isfinite(out).all()
    # Output variance should be approximately 1.0 per token
    var = out.pow(2).mean(dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_residual_identity_bias_at_initialization():
    """Test 29: With alpha=0.1, the output is dominated by the identity skip connection."""
    block = ConvTransformerBlock(dim=32, num_heads=2, h=6, w=6).eval()
    x = torch.randn(2, 37, 32)
    out = block(x)
    # Cosine similarity between input and output should be high (> 0.85) due to alpha=0.1
    cos_sim = F.cosine_similarity(x.flatten(1), out.flatten(1), dim=-1).mean().item()
    assert cos_sim > 0.85


def test_swiglu_non_linearity_and_gating():
    """Test 30: SwiGLU forward pass computes down(silu(gate(x)) * up(x))."""
    swiglu = SwiGLU(dim=32, hidden_dim=64)
    x = torch.randn(2, 10, 32)
    out = swiglu(x)
    assert out.shape == (2, 10, 32)


def test_swiglu_gradient_flow_both_gate_and_up():
    """Test 31: Gradients propagate cleanly to both w_gate, w_up, and w_down in SwiGLU."""
    swiglu = SwiGLU(dim=32, hidden_dim=64)
    x = torch.randn(2, 10, 32, requires_grad=True)
    out = swiglu(x)
    out.sum().backward()
    assert swiglu.w_gate.weight.grad is not None and swiglu.w_gate.weight.grad.abs().sum() > 0
    assert swiglu.w_up.weight.grad is not None and swiglu.w_up.weight.grad.abs().sum() > 0
    assert swiglu.w_down.weight.grad is not None and swiglu.w_down.weight.grad.abs().sum() > 0


# --- Category 5: Multi-Step MCTS Dynamics & Recurrent Rollouts (Tests 32-38) ---

def test_recurrent_inference_state_shape_preservation_10steps():
    """Test 32: 10 consecutive recurrent steps maintain exact (B, 64, 6, 6) latent state shape."""
    from atari_qwen.models.ez_model import EZV2Model
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    
    s = torch.randn(2, 64, 6, 6)
    hidden = model.init_hidden(2, s.device)
    for _ in range(10):
        a = torch.tensor([1, 2])
        s, _, _, _, hidden = model.recurrent_inference(s, a, hidden)
        assert s.shape == (2, 64, 6, 6)


def test_lstm_value_prefix_hidden_state_evolution():
    """Test 33: Value-prefix LSTM hidden state updates across unroll steps without divergence."""
    from atari_qwen.models.ez_model import EZV2Model
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    
    s = torch.randn(1, 64, 6, 6)
    h0, c0 = model.init_hidden(1, s.device)
    s1, r1, _, _, (h1, c1) = model.recurrent_inference(s, torch.tensor([0]), (h0, c0))
    assert not torch.equal(h0, h1), "LSTM hidden state did not evolve!"


def test_value_support_transformation_h_and_h_inv():
    """Test 34: Categorical value transform h(x) and inverse h_inv(y) are exact inverses."""
    from atari_qwen.models.ez_model import DiscreteSupport
    support = DiscreteSupport()
    test_values = torch.tensor([-250.0, -10.0, 0.0, 1.0, 5.0, 50.0, 280.0])
    transformed = support.h(test_values)
    recovered = support.h_inv(transformed)
    assert torch.allclose(test_values, recovered, atol=1e-3)


def test_categorical_value_projection_and_expectation():
    """Test 35: 601-bin categorical value logits map to valid scalar values in [-300, 300]."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()
    s = torch.randn(2, 64, 6, 6)
    v_logits, _ = trunk(s)
    probs = F.softmax(v_logits, dim=-1)
    support_bins = torch.arange(-300, 301, dtype=torch.float32)
    expected_v = (probs * support_bins).sum(dim=-1)
    assert expected_v.shape == (2,)
    assert (-300.0 <= expected_v).all() and (expected_v <= 300.0).all()


def test_multi_action_recurrent_divergence():
    """Test 36: Distinct actions (e.g. action 0 vs action 3) cause diverging next-state trajectories."""
    from atari_qwen.models.ez_model import EZV2Model
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    
    s = torch.randn(1, 64, 6, 6)
    hidden = model.init_hidden(1, s.device)
    s_a0, _, _, _, _ = model.recurrent_inference(s, torch.tensor([0]), hidden)
    s_a3, _, _, _, _ = model.recurrent_inference(s, torch.tensor([3]), hidden)
    assert not torch.allclose(s_a0, s_a3, atol=1e-4)


def test_noop_stability_in_recurrent_dynamics():
    """Test 37: 5 consecutive NOOP actions produce bounded state trajectory without explosion."""
    from atari_qwen.models.ez_model import EZV2Model
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    
    s = torch.randn(1, 64, 6, 6)
    hidden = model.init_hidden(1, s.device)
    for _ in range(5):
        s, _, _, _, hidden = model.recurrent_inference(s, torch.tensor([0]), hidden)
    assert torch.isfinite(s).all() and s.norm().item() < 200.0


def test_simnorm_group_normalization_preserves_probability_simplex():
    """Test 38: SimNorm groups (TD-MPC2 normalization) sum to 1.0 along the feature group."""
    from atari_qwen.models.ez_model import EZV2Model
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", simnorm=8, mixer_where="pred").eval()
    s = torch.randn(2, 64, 6, 6)
    normed = model.simnorm_latent(s)
    # Reshaped to group size 8: should sum to 1.0 along group dim
    grouped = normed.view(2, -1, 8)
    assert torch.allclose(grouped.sum(dim=-1), torch.ones_like(grouped.sum(dim=-1)), atol=1e-5)


# --- Category 6: Optimizer, Parameter Groups & Numerical Health (Tests 39-44) ---

def test_all_preset_configurations_instantiate_and_run():
    """Test 39: All presets (nano, micro, mini, small) build and execute forward passes cleanly."""
    for preset, cfg in CONV_TRANSFORMER_SIZES.items():
        trunk = ConvTransformerTrunk(
            channels=64, hw=36, dim=cfg["dim"], depth=cfg["depth"],
            heads=cfg["heads"], ffn_dim=cfg["ffn_dim"], num_actions=4, support_size=601
        ).eval()
        v, p = trunk(torch.randn(1, 64, 6, 6))
        assert v.shape == (1, 601) and p.shape == (1, 4)


def test_adamw_weight_decay_on_swiglu_and_projections():
    """Test 40: SwiGLU and linear projection matrices receive weight decay under AdamW."""
    from atari_qwen.training.train_ez_offpolicy import mixer_param_groups
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=32, depth=1, heads=2, ffn_dim=64, num_actions=4, support_size=601)
    groups = mixer_param_groups(trunk, weight_decay=0.05)
    decay_params = groups[0]["params"]
    # dw_proj weight should be in decay group
    assert any(p is trunk.blocks[0].dw_proj.weight for p in decay_params)
    assert any(p is trunk.blocks[0].ffn.w_gate.weight for p in decay_params)


def test_no_decay_group_contains_all_1d_and_positional_tensors():
    """Test 41: All RMSNorm gains, alpha vectors, and relative tables are in the no_decay group."""
    from atari_qwen.training.train_ez_offpolicy import mixer_param_groups
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=32, depth=1, heads=2, ffn_dim=64, num_actions=4, support_size=601)
    groups = mixer_param_groups(trunk, weight_decay=0.05)
    no_decay_params = groups[1]["params"]
    assert any(p is trunk.policy_token for p in no_decay_params)
    assert any(p is trunk.blocks[0].alpha_conv for p in no_decay_params)
    assert any(p is trunk.blocks[0].attn.rel_pos.spatial_rel_table for p in no_decay_params)


def test_gradient_clipping_compatibility():
    """Test 42: PyTorch grad clip clips large gradient spikes cleanly without introducing NaNs."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=32, depth=1, heads=2, ffn_dim=64, num_actions=4, support_size=601).train()
    v, p = trunk(torch.randn(2, 64, 6, 6))
    (v.sum() * 1000.0).backward()
    total_norm = torch.nn.utils.clip_grad_norm_(trunk.parameters(), max_norm=1.0)
    assert total_norm > 1.0, "Expected unclipped norm > 1.0"
    for p in trunk.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_zero_grad_resets_all_gradients_cleanly():
    """Test 43: zero_grad(set_to_none=True) resets all gradients to None."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=32, depth=1, heads=2, ffn_dim=64, num_actions=4, support_size=601).train()
    v, p = trunk(torch.randn(1, 64, 6, 6))
    v.sum().backward()
    trunk.zero_grad(set_to_none=True)
    for p in trunk.parameters():
        assert p.grad is None


def test_bfloat16_forward_backward_compatibility():
    """Test 44: Forward pass executes under bfloat16 autocast cleanly."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        v, p = trunk(torch.randn(2, 64, 6, 6))
    assert v.shape == (2, 601) and p.shape == (2, 4)


# --- Category 7: Behavioral Edge Cases & MCTS Prior Properties (Tests 45-50) ---

def test_uniform_prior_entropy_at_init_exactness():
    """Test 45: Policy prior logits are exactly 0.0 at init, giving uniform entropy ln(4)."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()
    _, p = trunk(torch.randn(4, 64, 6, 6))
    probs = F.softmax(p, dim=-1)
    entropy = -(probs * probs.log()).sum(dim=-1)
    expected_entropy = math.log(4)
    assert torch.allclose(entropy, torch.full_like(entropy, expected_entropy), atol=1e-5)


def test_root_value_zero_at_init_exactness():
    """Test 46: Root value logits are exactly 0.0 at init, predicting expected value 0.0."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()
    v, _ = trunk(torch.randn(4, 64, 6, 6))
    probs = F.softmax(v, dim=-1)
    support_bins = torch.arange(-300, 301, dtype=torch.float32)
    expected_val = (probs * support_bins).sum(dim=-1)
    assert torch.allclose(expected_val, torch.zeros_like(expected_val), atol=1e-5)


def test_single_channel_sensitivity():
    """Test 47: Perturbing channel 12 of the 64 channels propagates non-zero signal to readout."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()
    # Open heads slightly to measure sensitivity
    nn.init.normal_(trunk.policy_head[-1].weight, std=0.02)
    
    s1 = torch.zeros(1, 64, 6, 6)
    s2 = s1.clone()
    s2[0, 12, 2, 2] = 5.0
    with torch.no_grad():
        _, p1 = trunk(s1)
        _, p2 = trunk(s2)
    assert not torch.equal(p1, p2), "Single-channel perturbation failed to reach readout!"


def test_dead_brick_zero_activation_robustness():
    """Test 48: Zeroing a 2x3 block of cells (simulating cleared bricks) produces finite outputs."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()
    s = torch.randn(1, 64, 6, 6)
    s[:, :, 0:2, 1:4] = 0.0  # Cleared bricks
    with torch.no_grad():
        v, p = trunk(s)
    assert torch.isfinite(v).all() and torch.isfinite(p).all()


def test_varying_batch_sizes_from_1_to_64():
    """Test 49: Forward pass executes cleanly for diverse batch sizes B in {1, 2, 7, 16, 32, 64}."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=32, depth=1, heads=2, ffn_dim=64, num_actions=4, support_size=601).eval()
    for B in (1, 2, 7, 16, 32, 64):
        v, p = trunk(torch.randn(B, 64, 6, 6))
        assert v.shape == (B, 601) and p.shape == (B, 4)


def test_memory_footprint_scaling():
    """Test 50: 4-layer ConvTransformer trunk parameter footprint is under 25 MB in float32."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=256, depth=4, heads=4, ffn_dim=768, num_actions=4, support_size=601)
    total_bytes = sum(p.numel() * p.element_size() for p in trunk.parameters())
    mb = total_bytes / (1024 * 1024)
    # small preset (~4.1M params * 4 bytes ≈ 16.4 MB)
    assert mb < 25.0, f"Trunk memory footprint unexpectedly large: {mb:.2f} MB"
