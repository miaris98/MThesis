"""Module-level regression tests for atari_qwen architecture components.

See docs/todo/TODO_GTRXL_COLLAPSE_EXPERIMENTS.md, "Module-level regression tests" section
(added after the S-024..S-043 input-invariance investigation). These are fast structural
sanity checks -- forward/backward passes produce input-dependent, gradient-bearing output --
run in seconds via pytest instead of discovering a broken module through a multi-week,
GPU-hour RL experiment campaign.

Non-goal: these tests check that a module is *structurally capable* of producing
input-dependent, gradient-bearing output. They are NOT a substitute for the actual RL
training-dynamics experiments in the collapse-investigation doc. The root cause found in
S-043 (ImpalaCNNEncoder's missing weight init) breaks the *training dynamics*, not a single
forward/backward pass -- every test in this file would have still passed throughout
S-024-S-042 on the broken encoder. That is expected and consistent with the design.
"""
import torch
import pytest

from atari_qwen.models.visual_encoders import NatureCNNEncoder, ImpalaCNNEncoder, PatchTokenizer
from atari_qwen.models.gtrxl_layers import GTrXLBlock
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic
from atari_qwen.models.efficientzero_v2_predictor import EfficientZeroV2Predictor

torch.manual_seed(0)


def _synthetic_inputs(batch: int = 2, channels: int = 4, size: int = 84):
    """Maximally-different synthetic observations, matching the E1/E2 layer probe inputs
    (atari_qwen/scripts/probe_layerwise_variance.py)."""
    zeros = torch.zeros(batch, channels, size, size)
    full = torch.full((batch, channels, size, size), 255.0)
    noise_a = torch.rand(batch, channels, size, size) * 255.0
    noise_b = torch.rand(batch, channels, size, size) * 255.0
    return [zeros, full, noise_a, noise_b]


def _rel_std(tensors):
    """Across-input relative std: std across the input-variant axis, normalized by the
    mean magnitude of the outputs. ~0 means the module ignores its input."""
    stacked = torch.stack([t.detach().flatten() for t in tensors], dim=0)  # (variants, features)
    std = stacked.std(dim=0)
    scale = stacked.abs().mean(dim=0).clamp_min(1e-6)
    return (std / scale).mean().item()


def _assert_all_params_have_gradient(module, prefix: str = ""):
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        assert p.grad is not None, f"{full_name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{full_name} gradient contains NaN/Inf"
        assert p.grad.abs().sum() > 0, f"{full_name} gradient is all-zero"


class TestVisualEncoders:
    """E1/E2 layer probe as an assertion: each encoder's output must vary across
    maximally-different inputs, not collapse to a near-constant embedding, and every
    parameter must receive a non-zero gradient."""

    ENCODER_CONFIGS = [
        (NatureCNNEncoder, {}),
        (ImpalaCNNEncoder, {}),
        (ImpalaCNNEncoder, dict(kaiming_init=True)),
        (ImpalaCNNEncoder, dict(orthogonal_init=True)),
        (PatchTokenizer, {}),
    ]

    @pytest.mark.parametrize("encoder_cls,kwargs", ENCODER_CONFIGS)
    def test_output_varies_across_inputs(self, encoder_cls, kwargs):
        encoder = encoder_cls(in_channels=4, embed_dim=64, **kwargs)
        encoder.eval()
        with torch.no_grad():
            outputs = [encoder(x) for x in _synthetic_inputs()]
        rel_std = _rel_std(outputs)
        assert rel_std > 1e-4, (
            f"{encoder_cls.__name__}{kwargs}: output rel_std={rel_std:.6f} across maximally "
            "different inputs -- suspiciously close to input-invariant."
        )

    @pytest.mark.parametrize("encoder_cls,kwargs", ENCODER_CONFIGS)
    def test_gradient_reaches_all_parameters(self, encoder_cls, kwargs):
        encoder = encoder_cls(in_channels=4, embed_dim=64, **kwargs)
        x = torch.rand(2, 4, 84, 84) * 255.0
        out = encoder(x)
        out.sum().backward()
        _assert_all_params_have_gradient(encoder, encoder_cls.__name__)

    def test_cross_architecture_parity(self):
        """All encoder variants should clear the same input-variance bar side by side --
        a newly added variant should not be silently worse than the ones already in use."""
        results = {}
        for encoder_cls, kwargs in self.ENCODER_CONFIGS:
            encoder = encoder_cls(in_channels=4, embed_dim=64, **kwargs)
            encoder.eval()
            with torch.no_grad():
                outputs = [encoder(x) for x in _synthetic_inputs()]
            label = f"{encoder_cls.__name__}{kwargs}"
            results[label] = _rel_std(outputs)
        assert all(v > 1e-4 for v in results.values()), results


class TestGTrXLBlock:
    """GRU gating (Parisotto et al.) must not silently degrade into a no-op, and every
    parameter must receive gradient (catches a gate defaulting to identity, as bg_init
    effectively did pre-S-035)."""

    def test_gating_vs_plain_residual_differ(self):
        dim = 32
        torch.manual_seed(1)
        gated = GTrXLBlock(dim=dim, num_heads=4, ffn_dim=64, use_gru_gating=True, bg_init=0.0)
        torch.manual_seed(1)
        plain = GTrXLBlock(dim=dim, num_heads=4, ffn_dim=64, use_gru_gating=False)

        x = torch.randn(2, 5, dim)
        out_gated = gated(x)
        out_plain = plain(x)
        diff = (out_gated - out_plain).abs().mean().item()
        assert diff > 1e-6, (
            "GRU-gated block produced output identical to a plain-residual block with the "
            "same weights -- the gate may have collapsed to a no-op."
        )

    @pytest.mark.parametrize("use_gru_gating", [True, False])
    def test_gradient_reaches_all_parameters(self, use_gru_gating):
        dim = 32
        block = GTrXLBlock(dim=dim, num_heads=4, ffn_dim=64, use_gru_gating=use_gru_gating)
        x = torch.randn(2, 5, dim, requires_grad=True)
        out = block(x)
        out.sum().backward()
        _assert_all_params_have_gradient(block, f"GTrXLBlock(gru={use_gru_gating})")

    def test_gate_bias_controls_initial_openness(self):
        """S-035: bg_init sets how close to identity the gate starts. A higher bg_init
        should keep the block's output closer to its input than a lower bg_init, for
        otherwise-identical weights."""
        dim = 32
        torch.manual_seed(2)
        high_bg = GTrXLBlock(dim=dim, num_heads=4, ffn_dim=64, use_gru_gating=True, bg_init=2.0)
        torch.manual_seed(2)
        low_bg = GTrXLBlock(dim=dim, num_heads=4, ffn_dim=64, use_gru_gating=True, bg_init=0.0)

        x = torch.randn(2, 5, dim)
        dist_high = (high_bg(x) - x).abs().mean().item()
        dist_low = (low_bg(x) - x).abs().mean().item()
        assert dist_high < dist_low, (
            f"bg_init=2.0 should stay closer to identity than bg_init=0.0 at init "
            f"(high_bg_dist={dist_high:.4f}, low_bg_dist={dist_low:.4f})"
        )


class TestActorCriticGradientBalance:
    """S-038: actor-head init compounding gain=0.01 across BOTH linear layers left the
    actor's gradient into the shared trunk ~65x-100x weaker than the critic's, which
    (combined with other factors) contributed to input-invariance. Assert the ratio stays
    within a sane band for every architecture that implements this actor/critic-head pattern."""

    @staticmethod
    def _grad_norm(module):
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.pow(2).sum().item()
        return total ** 0.5

    def test_gtrxl_agent_actor_critic_grad_ratio(self):
        torch.manual_seed(0)
        agent = ImpalaGTrXLAgent(
            action_dim=4, in_channels=4, embed_dim=32, depth=1,
            num_heads=2, ffn_dim=64, use_gru_gating=False,
        )
        x = torch.rand(2, 4, 84, 84) * 255.0
        logits, value, _ = agent(x)

        agent.zero_grad()
        logits.sum().backward(retain_graph=True)
        actor_grad = self._grad_norm(agent.actor_head)

        agent.zero_grad()
        value.sum().backward()
        critic_grad = self._grad_norm(agent.critic_head)

        assert actor_grad > 0 and critic_grad > 0
        ratio = actor_grad / critic_grad
        assert 0.1 < ratio < 10.0, (
            f"ImpalaGTrXLAgent actor/critic head gradient ratio={ratio:.4f} is outside the "
            f"sane 0.1x-10x band (actor={actor_grad:.6f}, critic={critic_grad:.6f}) -- would "
            "reproduce S-038's ~65x-100x actor/critic gradient asymmetry."
        )

    def test_qwen_actor_critic_grad_ratio(self):
        torch.manual_seed(0)
        agent = QwenAtariActorCritic(
            action_dim=4, in_channels=4, preset="tiny", encoder_type="nature_cnn",
            depth=1, embed_dim=32, num_heads=2, ffn_dim=64,
        )
        x = torch.rand(2, 4, 84, 84) * 255.0
        actor_repr, critic_repr = agent._forward_transformer(x)
        logits = agent.actor_head(actor_repr)
        value = agent.critic_head(critic_repr)

        agent.zero_grad()
        logits.sum().backward(retain_graph=True)
        actor_grad = self._grad_norm(agent.actor_head)

        agent.zero_grad()
        value.sum().backward()
        critic_grad = self._grad_norm(agent.critic_head)

        assert actor_grad > 0 and critic_grad > 0
        ratio = actor_grad / critic_grad
        assert 0.1 < ratio < 10.0, (
            f"QwenAtariActorCritic actor/critic head gradient ratio={ratio:.4f} is outside "
            f"the sane 0.1x-10x band (actor={actor_grad:.6f}, critic={critic_grad:.6f})."
        )


class TestEfficientZeroV2Predictor:
    """S-036: the SimSiam consistency loss reached its trivial collapse minimum
    (cosine similarity -> -0.9999, i.e. the predictor and projector learn to perfectly
    anti-align regardless of input) within ~2,500 training steps in a real run. Assert
    a handful of optimization steps on random targets does NOT already reach that
    trivial minimum -- flags representation collapse before it shows up in a full run."""

    def test_consistency_loss_not_trivially_collapsed(self):
        torch.manual_seed(0)
        latent_dim = 32
        predictor = EfficientZeroV2Predictor(latent_dim=latent_dim, action_dim=4, unroll_steps=3, hidden_dim=64)
        opt = torch.optim.Adam(predictor.parameters(), lr=1e-3)

        z0 = torch.randn(8, latent_dim)
        actions = torch.randint(0, 4, (8, 3))
        target_proj = torch.randn(8, 3, latent_dim)  # independent random targets, not derived from z0

        losses = []
        for _ in range(20):
            opt.zero_grad()
            out = predictor.unroll_trajectory(z0, actions)
            step_losses = [
                EfficientZeroV2Predictor.compute_consistency_loss(out["projections"][k], target_proj[:, k])
                for k in range(3)
            ]
            loss = torch.stack(step_losses).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] > -0.999, (
            f"Consistency loss reached {losses[-1]:.6f} within 20 steps on independent random "
            "targets -- suspiciously close to SimSiam's trivial collapse minimum (-1.0), "
            "matching the representation-collapse pattern from S-036."
        )

    def test_dynamics_gradient_reaches_all_parameters(self):
        torch.manual_seed(0)
        predictor = EfficientZeroV2Predictor(latent_dim=32, action_dim=4, unroll_steps=3, hidden_dim=64)
        z0 = torch.randn(4, 32, requires_grad=True)
        actions = torch.randint(0, 4, (4, 3))
        out = predictor.unroll_trajectory(z0, actions)
        loss = (
            sum(r.sum() for r in out["rewards"])
            + sum(v.sum() for v in out["values"])
            + sum(p.sum() for p in out["projections"])
        )
        loss.backward()
        _assert_all_params_have_gradient(predictor, "EfficientZeroV2Predictor")
