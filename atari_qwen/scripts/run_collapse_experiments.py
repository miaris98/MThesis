"""Batch runner for the input-invariance experiment matrix (E5-E12).

See docs/todo/TODO_GTRXL_COLLAPSE_EXPERIMENTS.md. Each experiment is a short 15k-step on-policy
run from the S-037 baseline config, varying one or more of the suspected causes identified by the
E1/E2 layer-wise probe:

  * actor-head init  -- pre-fix, gain=0.01 compounded across BOTH actor_head layers, leaving the
    actor's gradient into the shared trunk ~65x weaker than the critic's
  * entropy bonus    -- with near-zero advantages it is the only consistent gradient, and it
    pushes the policy toward uniform-for-every-input (= input-invariance)
  * critic coupling  -- near-constant value targets under sparse reward flatten the shared trunk

After each run the checkpoint is probed for the headline metric: how much the logits actually
vary across maximally-different inputs.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOTAL_STEPS = 15000
BASE = dict(
    total_steps=TOTAL_STEPS,
    num_envs=16,
    num_steps=128,
    eval_interval_updates=3,
    reward_loss_weight=0.0,
    ez_value_loss_weight=0.0,
    consistency_loss_weight=0.0,
    use_gru_gating=False,
)

EXPERIMENTS = {
    # E5: the S-037 baseline, but with the actor-head init bug fixed.
    "E5_fix_actor_init": dict(BASE),

    # E6: E5 + stop value-loss gradients from reaching the shared trunk.
    "E6_detach_critic": dict(BASE, detach_critic=True),

    # E7: E5 + no entropy bonus at all -- the most direct test of entropy dominance.
    "E7_no_entropy": dict(BASE, ent_coef=0.0),

    # E8: E5 + entropy annealed down rather than switched off.
    "E8_anneal_entropy": dict(BASE, ent_coef=0.01, ent_coef_end=0.0005),

    # E9: E5 + 10x learning rate.
    "E9_lr_10x": dict(BASE, learning_rate=2.5e-3),

    # E10: all three fixes together.
    "E10_all_fixes": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3),

    # E11: all fixes + the EfficientZero v2 auxiliary losses switched back on.
    "E11_all_fixes_plus_ez2": dict(
        BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3,
        reward_loss_weight=1.0, ez_value_loss_weight=0.25, consistency_loss_weight=0.5,
    ),

    # E12: all fixes + GTrXL gating switched back on.
    "E12_all_fixes_plus_gating": dict(
        BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3, use_gru_gating=True,
    ),

    # Control: the pre-fix actor init, to confirm the A/B is measuring what we think.
    "E13_control_legacy_init": dict(BASE, legacy_actor_init=True),

    # ---- Round 2: pushing E10's winning recipe (detach_critic + ent_coef=0 + lr=1e-3) further ----

    # E14/E15: map LR sensitivity either side of E10's 1e-3, now that entropy/critic confounds
    # are controlled -- tests whether E9's "high LR alone is destructive" was about LR magnitude
    # or about LR combined with entropy/critic-coupling still being active.
    "E14_e10_lr5e-4": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=5e-4),
    "E15_e10_lr2e-3": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=2e-3),

    # E16/E17: isolate which EZ2 auxiliary loss is responsible for E11's reversion -- reward-only
    # vs. consistency-only (the SimSiam consistency loss hit -0.9999 within ~2500 steps in every
    # prior run that enabled it, the textbook signature of representation collapse).
    "E16_e10_reward_only": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3,
                                 reward_loss_weight=1.0),
    "E17_e10_consistency_only": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3,
                                      consistency_loss_weight=0.5),

    # E18: consistency loss at a much smaller weight, to see if a little is tolerable.
    "E18_e10_consistency_low": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3,
                                     consistency_loss_weight=0.1),

    # E19: cheaper partial critic-decoupling via vf_coef instead of a full detach.
    "E19_e10_vf_coef_low": dict(BASE, ent_coef=0.0, learning_rate=1e-3, vf_coef=0.1),

    # E20: does the E10 recipe's variance gain keep climbing given more budget?
    "E20_e10_40k_steps": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3,
                               total_steps=40000, eval_interval_updates=4),

    # E21: remove advantage normalization -- near-zero raw advantages divided by a tiny std can
    # itself inject high-variance noise into the policy gradient.
    "E21_e10_no_adv_norm": dict(BASE, detach_critic=True, ent_coef=0.0, learning_rate=1e-3,
                                 normalize_advantages=False),
}


def run_one(name: str, kwargs: dict, out_root: Path):
    log_dir = f"results/collapse_exp/{name}"
    kwargs = dict(kwargs, log_dir=log_dir)
    arg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    code = (
        "from atari_qwen.training.train_gtrxl_ez2_onpolicy import train_onpolicy_gtrxl_ez2\n"
        f"train_onpolicy_gtrxl_ez2({arg_str})\n"
    )
    log_path = out_root / f"{name}.log"
    print(f"\n{'=' * 90}\n>>> {name}\n>>> {arg_str}\n{'=' * 90}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run([sys.executable, "-c", code], stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f">>> {name} finished rc={proc.returncode} in {elapsed / 60:.1f} min -> {log_path}", flush=True)
    return log_path, kwargs


def probe_checkpoint(name: str, kwargs: dict, out_root: Path):
    log_dir = Path(REPO_ROOT) / "results" / "collapse_exp" / name
    ckpts = sorted(log_dir.glob("*/checkpoints/model_best.pt"))
    if not ckpts:
        print(f">>> {name}: no checkpoint found, skipping probe", flush=True)
        return None
    cmd = [
        sys.executable, str(REPO_ROOT / "atari_qwen" / "scripts" / "probe_layerwise_variance.py"),
        "--ckpt", str(ckpts[-1]),
    ]
    if not kwargs.get("use_gru_gating", True):
        cmd.append("--no-gru-gating")
    probe_path = out_root / f"{name}.probe.log"
    with open(probe_path, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return probe_path


def extract_metrics(log_path: Path, probe_path: Path):
    out = {}
    if log_path and log_path.exists():
        text = log_path.read_text(errors="ignore")
        spreads = [
            float(line.split("LogitSpread:")[1].split("|")[0])
            for line in text.splitlines() if "LogitSpread:" in line
        ]
        advs = [
            float(line.split("RawAdv:")[1].split("|")[0])
            for line in text.splitlines() if "RawAdv:" in line
        ]
        ents = [
            float(line.split("Ent:")[1].split("|")[0])
            for line in text.splitlines() if "Ent:" in line
        ]
        scores = [
            float(line.split("Score:")[1].split("+/-")[0])
            for line in text.splitlines()
            if line.strip().startswith("[EVALUATION]") and "Score:" in line and "+/-" in line
        ]
        out["logit_spread_first"] = spreads[0] if spreads else None
        out["logit_spread_last"] = spreads[-1] if spreads else None
        out["raw_adv_last"] = advs[-1] if advs else None
        out["entropy_last"] = ents[-1] if ents else None
        out["scores"] = scores
        out["best_score"] = max(scores) if scores else None
    if probe_path and probe_path.exists():
        for line in probe_path.read_text(errors="ignore").splitlines():
            if line.startswith("12_logits"):
                parts = {p.split("=")[0]: float(p.split("=")[1])
                         for p in line.split()[1:] if "=" in p}
                out["probe_logit_rel_std"] = parts.get("rel_std")
                out["probe_logit_abs_std"] = parts.get("abs_std")
            if line.startswith("10_policy_repr"):
                parts = {p.split("=")[0]: float(p.split("=")[1])
                         for p in line.split()[1:] if "=" in p}
                out["probe_policy_repr_rel_std"] = parts.get("rel_std")
    return out


def main(only=None):
    out_root = Path(REPO_ROOT) / "results" / "collapse_exp" / "_logs"
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    names = [n for n in EXPERIMENTS if (only is None or n in only)]
    print(f"--> Running {len(names)} experiments: {names}", flush=True)

    for name in names:
        kwargs = EXPERIMENTS[name]
        try:
            log_path, used = run_one(name, kwargs, out_root)
            probe_path = probe_checkpoint(name, used, out_root)
            results[name] = extract_metrics(log_path, probe_path)
            print(f">>> {name} metrics: {json.dumps(results[name])}", flush=True)
        except Exception as e:
            print(f">>> {name} FAILED during post-processing: {e!r} (training may have still "
                  f"completed -- see the per-experiment log)", flush=True)
            results[name] = {"error": repr(e)}
        (out_root / "summary.json").write_text(json.dumps(results, indent=2))

    print(f"\n\n{'=' * 110}\nSUMMARY (reference: random-init policy_repr rel_std=0.307, "
          f"logits rel_std=0.243; S-037 baseline logit abs_std~0.0003)\n{'=' * 110}")
    hdr = f"{'experiment':<30} {'logit_spread':<14} {'probe_rel_std':<15} {'policy_repr':<13} {'entropy':<9} {'rawadv':<10} {'best'}"
    print(hdr)
    print("-" * 110)
    for name, m in results.items():
        print(f"{name:<30} "
              f"{(m.get('logit_spread_last') or 0):<14.5f} "
              f"{(m.get('probe_logit_rel_std') or 0):<15.5f} "
              f"{(m.get('probe_policy_repr_rel_std') or 0):<13.5f} "
              f"{(m.get('entropy_last') or 0):<9.4f} "
              f"{(m.get('raw_adv_last') or 0):<10.5f} "
              f"{m.get('best_score')}")
    print(f"\n--> summary.json written to {out_root / 'summary.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    main(only=args.only)
