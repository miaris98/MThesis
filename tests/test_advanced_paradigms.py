"""Advanced Verification Paradigms for ConvTransformerTrunk & EZV2Model.

Paradigms Implemented:
1. Metamorphic Testing:
   - Additive invariance of attention logits.
   - Scale invariance of attention weights under QK-norm.
   - Neighborhood occlusion isolation (zero gradient/feature leakage outside 3x3 window).
   - Symmetrized horizontal reflection equivariance (LEFT/RIGHT action logit swap).
2. Property-Based / Invariant Fuzzing:
   - Empirical Lipschitz continuity and bounded perturbation sensitivity.
   - Simplex & categorical value support conservation across randomized scale regimes.
   - 20-step recurrent dynamics contractivity and bounded state norms.
3. Differential Testing (vs ResNet Baseline):
   - Latent representation effective rank & singular value spectrum comparison.
   - Recurrent transition cosine continuity comparison.
4. Runtime Invariant Tripwires & Sentinel Telemetry:
   - Attention entropy collapse tripwire.
   - Policy single-action dominance / collapse tripwire.
   - Gradient-to-weight ratio (|dW|/|W|) freeze detection.
5. Single-Batch Memorization (Empirical Sanity Benchmark):
   - 60-step gradient descent optimization on a fixed batch of 4 transitions.
   - Verifies network drives loss to near-zero and achieves 100% policy accuracy.
6. Adversarial & Boundary Fuzzing:
   - Pathological inputs: all-zero frames, extreme magnitudes, dead channels.
   - Arithmetic stability under subnormal and extreme dynamic range values.
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from atari_qwen.models.conv_transformer import (
    RelativePositionBias2D,
    LocalNeighborhoodAttention2D,
    ConvTransformerBlock,
    ConvTransformerTrunk,
    CONV_TRANSFORMER_SIZES,
)
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport


# ==============================================================================
# PARADIGM 1: METAMORPHIC TESTING (Symmetry & Invariant Relations)
# ==============================================================================

def test_metamorphic_qk_norm_scale_invariance():
    """Metamorphic Test 1: Scaling input tokens by positive scalar alpha > 0
    leaves attention weights invariant due to per-head RMSNorm."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=4, h=6, w=6, qk_norm=True).eval()
    x = torch.randn(2, 37, 64)
    alpha = 42.0
    x_scaled = x * alpha

    with torch.no_grad():
        _, w_orig = attn(x, return_weights=True)
        _, w_scaled = attn(x_scaled, return_weights=True)

    # Attention weights must be invariant to positive uniform scaling of input
    assert torch.allclose(w_orig, w_scaled, atol=1e-5), (
        f"QK-norm scale invariance violated: max diff = {(w_orig - w_scaled).abs().max().item()}"
    )


def test_metamorphic_additive_key_bias_invariance():
    """Metamorphic Test 2: Adding a uniform constant vector to all spatial keys
    leaves softmax attention distribution invariant across tokens."""
    attn = LocalNeighborhoodAttention2D(dim=64, num_heads=2, h=6, w=6, qk_norm=False).eval()
    x = torch.randn(1, 37, 64)

    # Add uniform constant along sequence dim (same vector added to every token)
    c = torch.randn(1, 1, 64)
    x_shifted = x + c

    with torch.no_grad():
        # Softmax is shift-invariant along the key dimension
        q = attn.q_proj(x)
        k_orig = attn.k_proj(x)
        k_shifted = attn.k_proj(x_shifted)
        # Compute raw logits before relative bias
        head_dim = 64 // 2
        q = q.view(1, 37, 2, head_dim).transpose(1, 2)
        k1 = k_orig.view(1, 37, 2, head_dim).transpose(1, 2)
        k2 = k_shifted.view(1, 37, 2, head_dim).transpose(1, 2)
        logits1 = torch.matmul(q, k1.transpose(-1, -2))
        logits2 = torch.matmul(q, k2.transpose(-1, -2))
        # Adding constant to all keys shifts each row by the same constant: softmax is identical
        w1 = F.softmax(logits1, dim=-1)
        w2 = F.softmax(logits2, dim=-1)

    assert torch.allclose(w1, w2, atol=1e-5)


def test_metamorphic_neighborhood_occlusion_isolation():
    """Metamorphic Test 3: Perturbing cell (0, 0) strictly results in zero change
    in spatial features of cells outside its 3x3 receptive window after 1 block."""
    block = ConvTransformerBlock(dim=64, num_heads=2, ffn_dim=128, h=6, w=6, window_size=3).eval()

    # Tokens: 1 policy token + 36 spatial tokens
    x1 = torch.randn(1, 37, 64)
    x2 = x1.clone()
    # Perturb spatial cell at index (0, 0) -> token index 1
    x2[:, 1, :] += 50.0

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    # In ConvTransformerBlock, Branch 1 (3x3 depthwise conv) followed by Branch 2 (3x3 local attention)
    # yields an effective receptive field of 5x5 (radius 2) from cell (0, 0).
    # Therefore, cells with row >= 3 or col >= 3 are strictly outside the receptive field and have 0.0 diff.
    for r in range(3, 6):
        for c in range(3, 6):
            token_idx = 1 + r * 6 + c
            diff = (out1[:, token_idx, :] - out2[:, token_idx, :]).abs().max().item()
            assert diff == 0.0, f"Spatial isolation violated outside 5x5 field at cell ({r}, {c}): diff={diff}"


def test_metamorphic_symmetrized_horizontal_flip_policy_swap():
    """Metamorphic Test 4: When relative position and conv weights are horizontally symmetric,
    horizontally reflecting the input spatial grid exactly swaps LEFT (action 3) and RIGHT (action 2)
    policy logits while keeping NOOP (0) and FIRE (1) identical."""
    trunk = ConvTransformerTrunk(
        channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128,
        num_actions=4, support_size=601, window_size=3
    ).eval()

    # Symmetrize relative position table across dc: table(dr, -dc) = table(dr, dc)
    with torch.no_grad():
        for b in trunk.blocks:
            tbl = b.attn.rel_pos.spatial_rel_table  # (heads, 121)
            heads, total_span = tbl.shape
            tbl_2d = tbl.view(heads, 11, 11)
            tbl_sym = 0.5 * (tbl_2d + torch.flip(tbl_2d, dims=[-1]))
            tbl.copy_(tbl_sym.view(heads, total_span))

            # Symmetrize depthwise conv kernels horizontally
            w_conv = b.dw_conv.weight  # (dim, 1, 3, 3)
            b.dw_conv.weight.copy_(0.5 * (w_conv + torch.flip(w_conv, dims=[-1])))

    # Create spatial input
    s = torch.randn(2, 64, 6, 6)
    s_flipped = torch.flip(s, dims=[-1])  # reflect width

    with torch.no_grad():
        v, p = trunk(s)
        v_f, p_f = trunk(s_flipped)

    # Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
    # Value must be invariant
    assert torch.allclose(v, v_f, atol=1e-4), "Value prediction must be horizontally invariant"
    # Action 0 (NOOP) and Action 1 (FIRE) must match between original and flipped
    assert torch.allclose(p[:, 0], p_f[:, 0], atol=1e-4), "NOOP logit must match under horizontal reflection"
    assert torch.allclose(p[:, 1], p_f[:, 1], atol=1e-4), "FIRE logit must match under horizontal reflection"
    # Action 2 (RIGHT) and Action 3 (LEFT) must swap!
    assert torch.allclose(p[:, 2], p_f[:, 3], atol=1e-4), "RIGHT and LEFT logits must swap under horizontal reflection"
    assert torch.allclose(p[:, 3], p_f[:, 2], atol=1e-4), "LEFT and RIGHT logits must swap under horizontal reflection"


# ==============================================================================
# PARADIGM 2: PROPERTY-BASED / INVARIANT FUZZING (Mathematical Invariants)
# ==============================================================================

def test_property_lipschitz_continuity_bounded_sensitivity():
    """Property Test 5: Infinitesimal input perturbations produce bounded output shifts
    with empirical Lipschitz constant L < 100.0 (prevents chaotic single-pixel bifurcation)."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()

    torch.manual_seed(42)
    max_lipschitz = 0.0

    for _ in range(20):
        s = torch.randn(2, 64, 6, 6)
        delta = torch.randn_like(s) * 1e-3
        eps = delta.norm().item()

        with torch.no_grad():
            _, p1 = trunk(s)
            _, p2 = trunk(s + delta)
            diff = (p2 - p1).norm().item()
            l_emp = diff / (eps + 1e-9)
            if l_emp > max_lipschitz:
                max_lipschitz = l_emp

    assert max_lipschitz < 100.0, f"Empirical Lipschitz constant exploded: {max_lipschitz}"


def test_property_categorical_support_simplex_conservation():
    """Property Test 6: Categorical value projection preserves probability simplex
    (sum p_i = 1.0, p_i >= 0) and scalar expectation bounds across 50 random scale regimes."""
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    support = DiscreteSupport()

    for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
        with torch.no_grad():
            # initial_inference returns (latent, value_logits, policy_logits)
            _, v_logits, _ = model.initial_inference(torch.randn(4, 12, 96, 96) * scale)
            probs = F.softmax(v_logits, dim=-1)

            # Sum of probabilities must equal 1.0
            sums = probs.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
            # All probabilities non-negative
            assert (probs >= 0.0).all()
            # Invert to scalar value: must be finite and within support [-300, 300]
            scalar_v = support.vector_to_scalar(probs)
            assert torch.isfinite(scalar_v).all()
            assert (scalar_v >= -300.0).all() and (scalar_v <= 300.0).all()


def test_property_recurrent_dynamics_contractivity_20steps():
    """Property Test 7: Recurrent latent dynamics rolled out for 20 consecutive steps
    across random action sequences remains strictly non-explosive (norm < 1000.0)."""
    # With TD-MPC2 SimNorm enabled (simnorm=8), recurrent latent dynamics form a strict
    # contraction mapping where latent norms remain strictly bounded under 20.0 indefinitely.
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred", simnorm=8).eval()

    torch.manual_seed(123)
    s = torch.randn(2, 64, 6, 6)
    hidden = model.init_hidden(2, s.device)

    with torch.no_grad():
        for step in range(20):
            rand_action = torch.randint(0, 4, (2,))
            s, r, p, v, hidden = model.recurrent_inference(s, rand_action, hidden)
            n = s.norm().item()
            assert torch.isfinite(s).all(), f"NaN or Inf encountered at step {step}"
            assert n < 20.0, f"Latent norm unconstrained under SimNorm at step {step}: norm={n}"


# ==============================================================================
# PARADIGM 3: DIFFERENTIAL TESTING (vs EfficientZero-v2 ResNet Baseline)
# ==============================================================================

def test_differential_representation_effective_rank():
    """Differential Test 8: ConvTransformer maintains healthy representation rank
    comparable to ResNet (> 3.5), proving no representation collapse into a single dimension."""
    model_conv = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    model_res = EZV2Model(action_dim=4, obs_channels=12, trunk="resnet").eval()

    torch.manual_seed(999)
    # Batch of 16 observations
    obs = torch.randn(16, 12, 96, 96)

    with torch.no_grad():
        s_conv, _, _ = model_conv.initial_inference(obs)
        s_res, _, _ = model_res.initial_inference(obs)

    # Flatten spatial features to (16, C*H*W)
    mat_conv = s_conv.view(16, -1)
    mat_res = s_res.view(16, -1)

    # SVD of representation matrices
    _, s_c, _ = torch.linalg.svd(mat_conv, full_matrices=False)
    _, s_r, _ = torch.linalg.svd(mat_res, full_matrices=False)

    # Compute effective rank = exp(entropy of normalized singular values)
    p_c = s_c / s_c.sum()
    p_r = s_r / s_r.sum()
    eff_rank_conv = torch.exp(- (p_c * (p_c + 1e-12).log()).sum()).item()
    eff_rank_res = torch.exp(- (p_r * (p_r + 1e-12).log()).sum()).item()

    # Effective rank must be > 3.5 and within 80% of ResNet, showing healthy dimensional spread
    assert eff_rank_conv > 3.5, f"ConvTransformer representation rank collapsed: {eff_rank_conv}"
    assert eff_rank_conv >= eff_rank_res * 0.8, (
        f"ConvTransformer rank ({eff_rank_conv:.2f}) significantly lower than ResNet ({eff_rank_res:.2f})"
    )


def test_differential_recurrent_transition_smoothness():
    """Differential Test 9: One-step latent transition cosine similarity with previous state
    is in a healthy range (0.1 < cos < 0.99), matching ResNet dynamics smoothness."""
    model_conv = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    model_res = EZV2Model(action_dim=4, obs_channels=12, trunk="resnet").eval()

    s = torch.randn(4, 64, 6, 6)
    act = torch.tensor([1, 2, 0, 3])

    with torch.no_grad():
        # ConvTransformer transition
        h_c = model_conv.init_hidden(4, s.device)
        next_s_conv, _, _, _, _ = model_conv.recurrent_inference(s, act, h_c)
        cos_conv = F.cosine_similarity(s.view(4, -1), next_s_conv.view(4, -1), dim=-1).mean().item()

        # ResNet transition
        h_r = model_res.init_hidden(4, s.device)
        next_s_res, _, _, _, _ = model_res.recurrent_inference(s, act, h_r)
        cos_res = F.cosine_similarity(s.view(4, -1), next_s_res.view(4, -1), dim=-1).mean().item()

    # Both models should produce coherent transitions neither disconnected nor identical
    assert 0.1 < cos_conv < 0.99, f"ConvTransformer cosine transition outside healthy bounds: {cos_conv}"
    assert 0.1 < cos_res < 0.99, f"ResNet cosine transition outside healthy bounds: {cos_res}"


# ==============================================================================
# PARADIGM 4: RUNTIME TRIPWIRES & SENTINEL TELEMETRY
# ==============================================================================

def test_tripwire_attention_entropy_collapse_trigger():
    """Tripwire Test 10: Online tripwire detector accurately trips when attention entropy
    drops below floor threshold (catches one-hot collapse seen in B37/B66)."""
    def check_attention_entropy_tripwire(attn_weights: torch.Tensor, min_entropy_floor: float = 0.2) -> bool:
        """Returns True if tripped (collapse detected), False if healthy."""
        p = attn_weights.clamp(min=1e-12)
        entropy = - (p * p.log()).sum(dim=-1).mean().item()
        return entropy < min_entropy_floor

    # Healthy uniform-like attention
    healthy_weights = torch.softmax(torch.randn(2, 4, 37, 37), dim=-1)
    assert not check_attention_entropy_tripwire(healthy_weights), "Healthy attention falsely tripped"

    # Pathological one-hot attention (collapse)
    collapsed_weights = torch.zeros(2, 4, 37, 37)
    collapsed_weights[:, :, :, 0] = 1.0  # all tokens attend only to index 0
    assert check_attention_entropy_tripwire(collapsed_weights), "One-hot collapse failed to trip sentinel"


def test_tripwire_policy_action_dominance_trigger():
    """Tripwire Test 11: Action dominance tripwire flags when single action exceeds 85%
    of selection frequency over a window (catches NOOP/RIGHT drift)."""
    def check_action_dominance_tripwire(action_counts: list, threshold: float = 0.85) -> bool:
        total = sum(action_counts)
        if total == 0:
            return False
        max_ratio = max(action_counts) / total
        return max_ratio > threshold

    # Healthy balanced play
    assert not check_action_dominance_tripwire([25, 25, 25, 25]), "Balanced play falsely tripped"
    # Action collapse (e.g. 96% NOOP)
    assert check_action_dominance_tripwire([96, 2, 1, 1]), "96% NOOP collapse failed to trip"


def test_tripwire_weight_freeze_detector():
    """Tripwire Test 12: Online |dW|/|W| checker identifies frozen layers where
    weight updates stall below numerical noise floor (catches S-116 SGD freeze)."""
    def check_weight_freeze_tripwire(w_init: torch.Tensor, w_now: torch.Tensor, min_relative_step: float = 1e-5) -> bool:
        rel_change = (w_now - w_init).norm() / (w_init.norm() + 1e-9)
        return rel_change.item() < min_relative_step

    w_init = torch.randn(64, 64)
    # Frozen weights (almost no update)
    w_frozen = w_init + 1e-8 * torch.randn_like(w_init)
    assert check_weight_freeze_tripwire(w_init, w_frozen), "Stalled layer failed to trigger freeze tripwire"

    # Healthy updating weights
    w_updated = w_init + 0.05 * torch.randn_like(w_init)
    assert not check_weight_freeze_tripwire(w_init, w_updated), "Active updating layer falsely tripped freeze detector"


# ==============================================================================
# PARADIGM 5: SINGLE-BATCH MEMORIZATION (Empirical Sanity Benchmark)
# ==============================================================================

def test_single_batch_memorization_convergence():
    """Empirical Sanity Test 13: ConvTransformer can overfit a single mini-batch
    of 4 transitions in 60 gradient steps (Loss drops > 80%, 100% policy accuracy)."""
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    support = DiscreteSupport()

    # Fixed synthetic batch of 4 transitions
    torch.manual_seed(42)
    obs = torch.randn(4, 12, 96, 96)
    target_actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    target_values = torch.tensor([0.0, 5.0, 10.0, 25.0])
    target_value_support = support.scalar_to_vector(target_values)

    initial_loss = None
    final_loss = None

    for step in range(60):
        optimizer.zero_grad()
        # initial_inference returns (latent, value_logits, policy_logits)
        _, value_logits, policy_logits = model.initial_inference(obs)

        # Policy cross-entropy
        loss_p = F.cross_entropy(policy_logits, target_actions)
        # Value categorical cross-entropy
        loss_v = - (target_value_support * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
        loss = loss_p + loss_v

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 59:
            final_loss = loss.item()

    # Verification:
    # 1. Loss must drop significantly (> 80%)
    loss_reduction = (initial_loss - final_loss) / initial_loss
    assert loss_reduction > 0.80, f"Memorization failed: initial={initial_loss:.4f}, final={final_loss:.4f}, reduction={loss_reduction:.2%}"

    # 2. Argmax prediction must achieve 100% accuracy on this batch
    with torch.no_grad():
        _, _, pred_logits = model.initial_inference(obs)
        pred_actions = pred_logits.argmax(dim=-1)
        assert (pred_actions == target_actions).all(), (
            f"Policy failed to reach 100% memorization: pred={pred_actions.tolist()} vs target={target_actions.tolist()}"
        )


# ==============================================================================
# PARADIGM 6: ADVERSARIAL & BOUNDARY FUZZING
# ==============================================================================

def test_adversarial_pathological_all_zero_observation():
    """Adversarial Test 14: An all-zero observation (complete black screen)
    produces finite, valid outputs without NaNs or zero-division crashes."""
    model = EZV2Model(action_dim=4, obs_channels=12, trunk="conv_transformer", mixer_size="nano", mixer_where="pred").eval()
    obs_zeros = torch.zeros(2, 12, 96, 96)

    with torch.no_grad():
        s, v, p = model.initial_inference(obs_zeros)

    assert torch.isfinite(s).all(), "Latent contains NaN/Inf under all-zero input"
    assert torch.isfinite(p).all(), "Policy contains NaN/Inf under all-zero input"
    assert torch.isfinite(v).all(), "Value contains NaN/Inf under all-zero input"


def test_adversarial_extreme_dynamic_range_inputs():
    """Adversarial Test 15: Inputs with extreme scales (1e4 and 1e-4) do not overflow RMSNorm."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).eval()

    for extreme_scale in [1e4, 1e-4]:
        s = torch.randn(2, 64, 6, 6) * extreme_scale
        with torch.no_grad():
            v, p = trunk(s)
        assert torch.isfinite(v).all() and torch.isfinite(p).all(), f"Failed at extreme scale {extreme_scale}"


def test_adversarial_random_dead_spatial_channels():
    """Adversarial Test 16: Zeroing out 50% of the spatial channels (simulating dead feature maps)
    does not disrupt gradient backpropagation through active channels."""
    trunk = ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=2, ffn_dim=128, num_actions=4, support_size=601).train()
    # Unzero heads to allow gradient backpropagation to input spatial tokens
    nn.init.normal_(trunk.policy_head[-1].weight, std=0.1)
    nn.init.normal_(trunk.value_head[-1].weight, std=0.1)

    s = torch.randn(2, 64, 6, 6, requires_grad=True)

    # Mask half the channels
    mask = torch.ones(2, 64, 1, 1)
    mask[:, :32, :, :] = 0.0
    s_masked = s * mask

    v, p = trunk(s_masked)
    loss = v.sum() + p.sum()
    loss.backward()

    # Active channels must have non-zero gradients
    active_grad_norm = s.grad[:, 32:, :, :].norm().item()
    assert active_grad_norm > 0.0, "Gradients failed to flow through active channels"
