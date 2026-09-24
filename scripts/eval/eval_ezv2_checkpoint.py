"""Evaluate one saved EfficientZero V2 checkpoint with the standard Atari protocol.

Run from the EfficientZeroV2 repo root with its venv:
    python eval_ezv2_checkpoint.py <model_N.p> <out.json> [n_episodes=10]

Uses the repo's own ez.eval.eval() and its config (ez/config/config.yaml + exp/atari_breakout.yaml)
unchanged. The repo's eval.py entry point hard-codes n_episodes=1, and its in-training evaluator
runs while training, so neither gives a checkpoint score at a fixed data budget - which is what the
S-055 comparison against our trainer needs (EZ-V2 at 10k/20k vs ours at 10k/20k).
"""
import json
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from ez import agents
from ez.eval import eval as ez_eval


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    config = OmegaConf.merge(OmegaConf.load("ez/config/config.yaml"),
                             OmegaConf.load("ez/config/exp/atari_breakout.yaml"))
    agent = agents.names[config.agent_name](config)
    ray.init(num_gpus=torch.cuda.device_count(), num_cpus=8, object_store_memory=4 * 1024 ** 3)
    model = agent.build_model()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    scores = np.asarray(ez_eval(agent, model, n_episodes, Path(out).with_suffix(""), config,
                                max_steps=27000, use_pb=False, verbose=0), dtype=np.float64)
    res = {"checkpoint": ckpt, "n_episodes": n_episodes, "max_steps": 27000,
           "scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()),
           "se": float(scores.std() / np.sqrt(len(scores)))}
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"EZV2_EVAL {Path(ckpt).name}: mean {res['mean']:.2f} +/- {res['std']:.2f} (SE {res['se']:.2f}) over {n_episodes}")


if __name__ == "__main__":
    main()
