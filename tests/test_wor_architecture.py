"""Architecture-level tests for the World on Rails decision heads.

These assert properties of the *architectures* rather than of any trained model, which
is why none of them needs a training run: a head that cannot represent something at
initialisation cannot learn it either. Each one corresponds to a defect found by
probing the first CNN-versus-transformer comparison - see
challenges/challenges_13_transformer_head_underperformance.md.
"""
import pytest
import torch

from src.models.world_on_rails import WorldOnRailsPolicy, QwenWorldOnRailsPolicy


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


def test_small_model_sizes_are_available_and_sized_right():
    """The offline dataset is ~9,600 frames; 100M/500M/900M are all on the wrong side
    of the size/data trade-off, so the sweep needs smaller trunks to compare against."""
    from src.models.world_on_rails.qwen_wor_policy import _MODEL_SIZES

    assert {"10m", "30m"} <= set(_MODEL_SIZES)

    for size, (low, high) in (("10m", (7e6, 15e6)), ("30m", (20e6, 40e6))):
        policy = QwenWorldOnRailsPolicy(
            backbone_name="resnet18", pretrained=False, model_size=size, vision_grid=4
        )
        trunk = sum(p.numel() for p in policy.trunk.parameters())
        assert low < trunk < high, f"{size} trunk is {trunk / 1e6:.1f}M, outside its name"
