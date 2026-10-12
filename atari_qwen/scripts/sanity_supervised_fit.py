"""E4: can ImpalaGTrXLAgent learn ANY input-dependent mapping at all?

Strips away RL entirely. Trains the network with plain cross-entropy on a trivially separable
task: all-black frames -> action 0, all-white frames -> action 1 (optionally two distinct real
gameplay frames instead, which is a harder but still trivial task). A functioning network should
solve this in tens of gradient steps.

If it CANNOT, there is a structural bug in the forward/backward path and no amount of RL tuning
or training budget will ever help -- which would explain every result in S-024 through S-037.
If it CAN, the architecture is sound and the problem is the RL signal/optimization balance.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent


def run(steps: int, lr: float, use_gru_gating: bool, fix_actor_init: bool, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    agent = ImpalaGTrXLAgent(
        action_dim=4, in_channels=4, embed_dim=256, depth=4,
        num_heads=4, ffn_dim=1024, unroll_steps=5, bg_init=0.0,
        use_gru_gating=use_gru_gating,
    ).to(device)

    if fix_actor_init:
        # Standard CleanRL practice: small gain ONLY on the output layer, sqrt(2) on hidden.
        linears = [m for m in agent.actor_head if isinstance(m, torch.nn.Linear)]
        for m in linears[:-1]:
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
        torch.nn.init.orthogonal_(linears[-1].weight, gain=0.01)
        torch.nn.init.zeros_(linears[-1].bias)
        print("--> Applied FIXED actor-head init (sqrt(2) hidden, 0.01 output only)")
    else:
        print("--> Using CURRENT actor-head init (gain=0.01 on EVERY layer)")

    print(f"--> use_gru_gating={use_gru_gating} | lr={lr} | steps={steps}")

    opt = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=0.0)
    agent.train()

    black = torch.zeros((1, 4, 84, 84), dtype=torch.uint8, device=device)
    white = torch.full((1, 4, 84, 84), 255, dtype=torch.uint8, device=device)

    print(f"\n{'step':<8} {'loss':<12} {'acc':<8} {'logit_gap(black-white)':<24} {'grad_norm_trunk'}")
    print("-" * 88)

    for step in range(steps + 1):
        half = batch_size // 2
        x = torch.cat([black.expand(half, -1, -1, -1), white.expand(half, -1, -1, -1)], dim=0)
        y = torch.cat([
            torch.zeros(half, dtype=torch.long, device=device),
            torch.ones(half, dtype=torch.long, device=device),
        ])

        logits, _, _ = agent(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad()
        loss.backward()

        trunk_grad = torch.sqrt(sum(
            (p.grad.detach() ** 2).sum()
            for p in agent.visual_encoder.parameters() if p.grad is not None
        )).item()

        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()

        if step % max(1, steps // 20) == 0 or step == steps:
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == y).float().mean().item()
                lb, _, _ = agent(black)
                lw, _, _ = agent(white)
                gap = (lb.squeeze(0) - lw.squeeze(0)).abs().max().item()
            print(f"{step:<8} {loss.item():<12.6f} {acc:<8.3f} {gap:<24.6f} {trunk_grad:.3e}")

    with torch.no_grad():
        lb, _, _ = agent(black)
        lw, _, _ = agent(white)
    print(f"\n--> final logits(black) = {[f'{v:.4f}' for v in lb.squeeze(0).tolist()]}")
    print(f"--> final logits(white) = {[f'{v:.4f}' for v in lw.squeeze(0).tolist()]}")
    gap = (lb - lw).abs().max().item()
    solved = lb.argmax(dim=-1).item() == 0 and lw.argmax(dim=-1).item() == 1
    print(f"--> max |logit difference| between the two classes: {gap:.6f}")
    if solved:
        print("--> VERDICT: SOLVED -- the architecture can learn input-dependent mappings.")
        print("    The forward/backward path is structurally fine; the problem lies in the RL")
        print("    signal / optimization balance, not the network itself.")
    else:
        print("--> VERDICT: FAILED -- the network cannot separate two maximally-different inputs.")
        print("    This is a structural bug; no RL tuning or step budget will ever fix it.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--no-gru-gating", action="store_true")
    parser.add_argument("--fix-actor-init", action="store_true")
    args = parser.parse_args()
    run(args.steps, args.lr, use_gru_gating=not args.no_gru_gating, fix_actor_init=args.fix_actor_init)
