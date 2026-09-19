"""Incremental-integration plan (struggle-solutions S-041): start from the PROVEN pipeline
(train_ppo.py + QwenAtariActorCritic, which genuinely climbed Breakout 0.00 -> 17.00 in S-029)
and swap in ImpalaGTrXLAgent's components one at a time, instead of continuing to vary
hyperparameters on a stack that already differs from the proven baseline in five places at once
(visual encoder, transformer block type, norm type, head init scheme, EfficientZero v2 losses).

Each step trains a short (15k-step) run with train_ppo.train(), then probes the resulting
checkpoint the same way probe_layerwise_variance.py does (same synthetic input battery, same
rel_std metric) so every step is directly comparable to every number in S-024-S-041.

Steps:
  I0_baseline          -- QwenAtariActorCritic exactly as train_ppo.py always ran it (nature_cnn
                           encoder, "qwen" blocks). The known-good reference, re-measured fresh.
  I1_impala_encoder     -- swap ONLY the visual encoder to ImpalaCNNEncoder (same one
                           ImpalaGTrXLAgent uses). Blocks/heads/trainer all stay proven.
  I2_gtrxl_blocks        -- I1 + swap the transformer block type to GTrXLBlock (GRU gating off,
                           matching ImpalaGTrXLAgent's use_gru_gating=False isolation config).
                           This is the component most suspected after S-035-S-041's GTrXL-centric
                           findings -- if learning breaks exactly here, it points squarely at the
                           GTrXL block's attention/FFN structure itself, not gating or auxiliary
                           losses (both already ruled out separately).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOTAL_TIMESTEPS = 15000  # matches every 15k-step experiment in E5-E51 for direct comparability

STEPS = {
    "I0_baseline": dict(encoder_type="nature_cnn", block_type="qwen"),
    "I1_impala_encoder": dict(encoder_type="impala_cnn", block_type="qwen"),
    "I2_gtrxl_blocks": dict(encoder_type="impala_cnn", block_type="gtrxl"),
}

# S-042 follow-up: I0_baseline at the 15k budget above FAILS its own diagnostic
# (probe_logit_rel_std=0.013, flat 0.0 eval) -- because S-029's own genuinely-successful run of
# this exact pipeline stayed at 0.0 until ~150k steps and only reached 17.0 by 299k
# (struggle-solutions S-029: 0.00@51k, 0.00@100k, 12.00@151k, 12.00@200k, 16.00@251k, 17.00@299k).
# 15k steps was never long enough to distinguish "broken" from "hasn't gotten there yet" for ANY
# of these architectures. Re-run all three steps at S-029's actual 300k-step budget, same eval
# cadence, to get the real apples-to-apples reference trajectory.
LONG_TOTAL_TIMESTEPS = 300000
LONG_STEPS = {
    "I0_baseline_300k": dict(encoder_type="nature_cnn", block_type="qwen"),
    "I1_impala_encoder_300k": dict(encoder_type="impala_cnn", block_type="qwen"),
    "I2_gtrxl_blocks_300k": dict(encoder_type="impala_cnn", block_type="gtrxl"),
    # S-043: I1 broke the clean climb into 0/11 oscillation -- ImpalaCNNEncoder has no explicit
    # weight init (unlike NatureCNNEncoder). Test whether giving it NatureCNNEncoder's own
    # kaiming_normal_ scheme recovers I0's clean trajectory.
    "I1b_impala_kaiming_init_300k": dict(encoder_type="impala_cnn", block_type="qwen",
                                          impala_kaiming_init=True),
}
STEPS.update(LONG_STEPS)


def run_one(name: str, overrides: dict, out_root: Path):
    log_dir = f"results/incremental_integration/{name}"
    total_steps = LONG_TOTAL_TIMESTEPS if name in LONG_STEPS else TOTAL_TIMESTEPS
    eval_freq = 50000 if name in LONG_STEPS else 3000
    eval_episodes = 5 if name in LONG_STEPS else 3
    train_code = (
        "from atari_qwen.config.atari_config import get_config\n"
        "from atari_qwen.training.train_ppo import train\n"
        f"config = get_config(total_timesteps={total_steps}, num_envs=16, num_steps=128, "
        f"eval_freq_steps={eval_freq}, eval_episodes={eval_episodes}, log_dir={log_dir!r}, "
        f"encoder_type={overrides['encoder_type']!r}, block_type={overrides['block_type']!r}, "
        f"impala_kaiming_init={overrides.get('impala_kaiming_init', False)!r})\n"
        "train(config)\n"
    )
    log_path = out_root / f"{name}.log"
    print(f"\n{'=' * 90}\n>>> {name}  {overrides}\n{'=' * 90}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run([sys.executable, "-c", train_code], stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f">>> {name} finished rc={proc.returncode} in {elapsed / 60:.1f} min -> {log_path}", flush=True)
    return log_path


def probe_checkpoint(name: str, overrides: dict, out_root: Path):
    ckpts = sorted((REPO_ROOT / "results" / "incremental_integration" / name).glob("*/checkpoints/model_best.pt"))
    if not ckpts:
        print(f">>> {name}: no checkpoint found, skipping probe", flush=True)
        return None
    probe_code = (
        "import torch\n"
        "from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic\n"
        "from atari_qwen.scripts.probe_layerwise_variance import build_inputs\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "batch, labels = build_inputs('BreakoutNoFrameskip-v4', device)\n"
        "model = QwenAtariActorCritic(action_dim=4, in_channels=4, preset='tiny', "
        f"encoder_type={overrides['encoder_type']!r}, block_type={overrides['block_type']!r}, "
        f"impala_kaiming_init={overrides.get('impala_kaiming_init', False)!r}).to(device)\n"
        f"ckpt = torch.load({str(ckpts[-1])!r}, map_location=device, weights_only=False)\n"
        "model.load_state_dict(ckpt.get('model_state_dict', ckpt))\n"
        "model.eval()\n"
        "with torch.no_grad():\n"
        "    actor_repr, _ = model._forward_transformer(batch)\n"
        "    logits = model.actor_head(actor_repr)\n"
        "flat = logits.reshape(logits.shape[0], -1).float()\n"
        "abs_std = flat.std(dim=0).mean().item()\n"
        "scale = flat.abs().mean().item()\n"
        "rel_std = abs_std / (scale + 1e-12)\n"
        "print(f'12_logits rel_std={rel_std:.6f} abs_std={abs_std:.6f} scale={scale:.6f}')\n"
    )
    probe_path = out_root / f"{name}.probe.log"
    with open(probe_path, "w") as fh:
        subprocess.run([sys.executable, "-c", probe_code], stdout=fh, stderr=subprocess.STDOUT)
    return probe_path


def extract_metrics(log_path: Path, probe_path: Path):
    out = {}
    if log_path and log_path.exists():
        text = log_path.read_text(errors="ignore")
        scores = [
            float(line.split("Mean Return:")[1].split("+/-")[0])
            for line in text.splitlines()
            if line.strip().startswith("[EVALUATION]") and "Mean Return:" in line
        ]
        out["scores"] = scores
        out["best_score"] = max(scores) if scores else None
    if probe_path and probe_path.exists():
        for line in probe_path.read_text(errors="ignore").splitlines():
            if line.startswith("12_logits"):
                parts = {p.split("=")[0]: float(p.split("=")[1]) for p in line.split()[1:] if "=" in p}
                out["probe_logit_rel_std"] = parts.get("rel_std")
                out["probe_logit_abs_std"] = parts.get("abs_std")
    return out


def main(only=None):
    out_root = Path(REPO_ROOT) / "results" / "incremental_integration" / "_logs"
    out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    names = [n for n in STEPS if (only is None or n in only)]
    print(f"--> Running {len(names)} incremental-integration steps: {names}", flush=True)

    for name in names:
        overrides = STEPS[name]
        try:
            log_path = run_one(name, overrides, out_root)
            probe_path = probe_checkpoint(name, overrides, out_root)
            results[name] = extract_metrics(log_path, probe_path)
            print(f">>> {name} metrics: {json.dumps(results[name])}", flush=True)
        except Exception as e:
            print(f">>> {name} FAILED during post-processing: {e!r}", flush=True)
            results[name] = {"error": repr(e)}
        (out_root / "summary.json").write_text(json.dumps(results, indent=2))

    print(f"\n\n{'=' * 100}\nSUMMARY (reference: E10's probe_logit_rel_std=0.03008; healthy random-init=0.243)\n{'=' * 100}")
    for name, m in results.items():
        print(f"{name:<22} probe_logit_rel_std={m.get('probe_logit_rel_std')} "
              f"scores={m.get('scores')} best={m.get('best_score')}")
    print(f"\n--> summary.json written to {out_root / 'summary.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    main(only=args.only)
