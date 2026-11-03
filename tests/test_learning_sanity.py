"""TODO (2026-11-03): learning sanity. The unit tests check the trunk at initialisation; none checks that it can learn. A broken gradient path
(a detached block, a zeroed head that never moves, a loss that does not reach the trunk) passes every other test. This trains the trunk on a
fixed batch with a target it can fit, and asserts that the loss falls and that the gradient reaches the blocks, not only the output head."""
import torch
import torch.nn.functional as F

from atari_qwen.models.conv_transformer import ConvTransformerTrunk


def _trunk(seed=0):
    torch.manual_seed(seed)
    return ConvTransformerTrunk(channels=64, hw=36, dim=64, depth=2, heads=4, ffn_dim=128, num_actions=4, support_size=601, qk_norm=True)


def test_the_policy_loss_falls_on_a_fixed_batch():
    trunk = _trunk()
    g = torch.Generator().manual_seed(1)
    s = torch.randn(32, 64, 6, 6, generator=g)
    target = torch.randint(0, 4, (32,), generator=g)
    opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3)
    losses = []
    for _ in range(150):
        _, logits = trunk(s)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < 0.5 * losses[0], f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_the_gradient_reaches_the_blocks_once_the_zero_init_head_has_moved():
    """The prediction heads start at zero (uniform prior), so at step 0 the loss has no gradient into the blocks: the output layer's gradient
    is zero, and so is everything behind it. That is by design, but it means the first update only moves the heads. After one step the
    gradient must reach every attention parameter, or the trunk could never learn."""
    trunk = _trunk()
    s = torch.randn(8, 64, 6, 6)
    target = torch.zeros(8, dtype=torch.long)
    opt = torch.optim.AdamW(trunk.parameters(), lr=1e-3)
    _, logits = trunk(s)
    F.cross_entropy(logits, target).backward()
    assert all(float(p.grad.abs().sum()) == 0.0 for n, p in trunk.named_parameters()
               if n.startswith("blocks.") and "attn" in n and p.grad is not None), "step-0 gradient should be zero behind a zero head"
    opt.step()
    opt.zero_grad()
    _, logits = trunk(s)
    F.cross_entropy(logits, target).backward()
    last = f"blocks.{len(trunk.blocks) - 1}."
    # structural, not a bug: only the [POLICY] row of the last block reaches the heads, so the biases that act on the spatial rows' queries
    # (spatial->spatial and spatial->policy) get no gradient there. Finding (2026-11-03): the last block's spatial rows are computed and unused.
    dead = {last + "attn.rel_pos.spatial_rel_table", last + "attn.rel_pos.spatial_to_pol_bias"}
    zero = {n for n, p in trunk.named_parameters() if n.startswith("blocks.") and "attn" in n and p.dim() > 1
            and (p.grad is None or float(p.grad.abs().sum()) == 0.0)}
    assert zero == dead, f"unexpected zero gradients: {zero ^ dead}"
    body = [p for n, p in trunk.named_parameters() if n.startswith("blocks.") and "attn" in n and p.dim() > 1 and n not in dead]
    assert all(float(p.grad.abs().sum()) > 0 for p in body), "no gradient in the attention blocks after one step"
