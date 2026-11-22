#!/usr/bin/env python3
"""Load-check a trained checkpoint the way an eval lane will: build the live agent, check that named weights were trained, drive two
frames. One copy of the check that a81_box.sh, a84_box.sh and arms_box.sh each carried inline (merged 2026-11-22).

    python scripts/eval/agent_smoke.py CKPT [--trained PATH ...] [--expect PATH=VALUE ...] [--env K=V ...]

  --trained PATH   the attribute under the agent's network must exist and have a non-zero sum (training reached it), e.g.
                   speed_moe.w2, reason_head.out.weight, trunk.moe.w2, vis_moe.w2
  --expect PATH=V  the attribute must equal V, compared as text, e.g. speed_moe.num_experts=4
  --env K=V        set before the agent is built, e.g. WOR_DANGER_GATE=theta=0.6
Prints one line (the control after a standing and a moving frame, plus the danger probability when the network has one); a failed
check exits 1 with the reason.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def resolve(obj, path):
    """getattr along a dotted path; None when any step is missing or None."""
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def check(net, trained=(), expect=()):
    """Return the first failed check as text, or None."""
    for path in trained:
        t = resolve(net, path)
        if t is None:
            return f"{path} missing: the agent rebuilt the model without it"
        if float(t.abs().sum()) <= 0:
            return f"{path} is still at its zero init: training did not reach it"
    for spec in expect:
        path, want = spec.split("=", 1)
        got = resolve(net, path)
        if str(got) != want:
            return f"{path} = {got}, expected {want}"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--trained", action="append", default=[])
    ap.add_argument("--expect", action="append", default=[])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--policy_arch", default="qwen30m")
    ap.add_argument("--backbone", default="regnety_032")
    a = ap.parse_args(argv)
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v

    import numpy as np
    from src.agents.wor_agent import WorldOnRailsAgent

    agent = WorldOnRailsAgent(checkpoint_path=a.checkpoint, policy_arch=a.policy_arch, backbone_name=a.backbone)
    failed = check(agent.net, a.trained, a.expect)
    if failed:
        sys.exit(failed)
    frame = np.random.default_rng(0).integers(0, 255, (512, 1024, 3), dtype=np.uint8)
    route = np.array([[2, 0], [4, 0], [6, 0], [8, 0]], np.float32)
    for speed in (0.0, 6.0):
        c = agent.run_step({"rgb_front": frame, "speed": speed, "command": 3, "route": route})
    debug = getattr(agent.net, "last_debug", None)
    danger = ""
    if isinstance(debug, dict) and "danger_p" in debug:
        danger = f" p_danger {debug['danger_p']} gate {getattr(agent.net, '_danger_gate', None) is not None}"
    print(f"control {c.steer:.3f} {c.throttle:.3f} {c.brake:.3f}{danger}")


if __name__ == "__main__":
    main()
