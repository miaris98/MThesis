"""TODO B33 step 0 (2026-10-16): depth-1 action aliasing in the EZ model.

Question: do two different root actions lead to the same next latent (e.g. FIRE ~ NOOP mid-rally)? If so, merging aliased root
actions frees Gumbel's slots and is simpler than a transposition table. Measured offline on real states: the checkpoint's own
policy plays the env (no search, sampled from its policy head), and at every visited state we compute the next latent
dynamics(s, a) for every action a and compare them pairwise by cosine similarity (the same 0.98 threshold B33 uses to verify a hash hit).

usage: python atari_qwen/training/b33_action_aliasing.py CKPT [CKPT ...] --steps 500 --envs 16 --out DIR
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from atari_qwen.envs.atari_wrappers import gym, make_vector_atari_envs
from atari_qwen.training.eval_ez_checkpoints import load_model, mlflow_run
from atari_qwen.training.train_ez_offpolicy import to_input

COS_ALIAS = 0.98
REL_ALIAS = 0.05  # relative distance between the two next latents: below 5% of the latent's norm = the same next latent
cos_per_root: list = []
rel_all: list = []


def aliasing_stats(model, obs_u8, device, meanings, counts_pairs, per_root):
    x = to_input(obs_u8, device)
    with torch.no_grad():
        s, _, p = model.initial_inference(x)
        probs = F.softmax(p, dim=-1)
        A = probs.shape[1]
        n = s.shape[0]
        # next latent for every (root, action): (n, A, C*H*W)
        s_rep = s.unsqueeze(1).expand(n, A, *s.shape[1:]).reshape(n * A, *s.shape[1:])
        a_rep = torch.arange(A, device=device).repeat(n)
        raw = model.dynamics(s_rep, a_rep).reshape(n, A, -1)
        iu = torch.triu_indices(A, A, offset=1, device=device)
        ra, rb = raw[:, iu[0]], raw[:, iu[1]]  # (n, pairs, D)
        # cosine (B33's hash-verify threshold): non-negative latents make it high even when they differ
        cos = F.cosine_similarity(ra, rb, dim=-1)  # (n, pairs)
        # relative distance: ||next_a - next_b|| / ||next_a||, the stricter test for "the same next latent"
        rel = (ra - rb).norm(dim=-1) / ra.norm(dim=-1).clamp_min(1e-8)
        alias_cos = cos >= COS_ALIAS
        alias = rel < REL_ALIAS
        sampled = torch.distributions.Categorical(probs=probs).sample()
    alias_per_root = alias.sum(dim=1).cpu().numpy()
    for r in range(n):
        per_root.append(int(alias_per_root[r]))
        cos_per_root.append(int(alias_cos[r].sum().item()))
    idx = alias.nonzero(as_tuple=False).cpu().numpy()
    pairs = iu.cpu().numpy()
    for r, k in idx:
        i, j = pairs[0][k], pairs[1][k]
        counts_pairs[(meanings[i], meanings[j])] += 1
    rel_all.append(rel.flatten().cpu().numpy())
    return sampled.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ck_path in args.ckpts:
        ck, a, model, _ = load_model(ck_path, device)
        model.eval()
        label = f"{ck_path.parent.parent.name}_{ck_path.stem}"
        probe = gym.make(a.env_id)
        meanings = probe.unwrapped.get_action_meanings()
        probe.close()
        envs = make_vector_atari_envs(a.env_id, num_envs=args.envs, seed=a.seed + 20_000, clip_reward=False, episodic_life=False,
                                      frame_size=a.frame_size, grayscale=False, fire_reset=False,
                                      max_episode_steps=27_000, repeat_action_probability=0.0)
        obs, _ = envs.reset(seed=a.seed + 20_000)
        counts_pairs, per_root = Counter(), []
        cos_per_root.clear(); rel_all.clear()
        for t in range(args.steps):
            act = aliasing_stats(model, obs, device, meanings, counts_pairs, per_root)
            obs, r, term, trunc, _ = envs.step(act)
            if np.logical_or(term, trunc).any():
                obs, _ = envs.reset()
        envs.close()
        per_root = np.array(per_root)
        cos_arr = np.array(cos_per_root)
        rel_flat = np.concatenate(rel_all)
        summ = {"roots": int(per_root.size),
                "frac_roots_with_alias_rel5pct": float((per_root > 0).mean()),
                "mean_aliased_pairs_per_root_rel5pct": float(per_root.mean()),
                "frac_roots_with_alias_cos098": float((cos_arr > 0).mean()),
                "median_rel_distance_all_pairs": float(np.median(rel_flat)), "actions": len(meanings),
                "rel_threshold": REL_ALIAS, "cos_threshold": COS_ALIAS}
        top = [{"pair": list(k), "count": int(v)} for k, v in counts_pairs.most_common(12)]
        out = {"checkpoint": str(ck_path), "label": label, "summary": summ, "top_alias_pairs": top}
        (args.out / f"{label}_b33.json").write_text(json.dumps(out, indent=1))
        mlf, run = mlflow_run(label, a, proto={"episodes": args.envs, "sims": 0, "sticky": 0.0, "tag": "b33_aliasing"},
                              tag="b33_aliasing")
        if mlf is not None:
            try:
                mlf.log_metrics({"frac_roots_with_alias_rel5pct": summ["frac_roots_with_alias_rel5pct"],
                                 "mean_aliased_pairs_per_root_rel5pct": summ["mean_aliased_pairs_per_root_rel5pct"],
                                 "frac_roots_with_alias_cos098": summ["frac_roots_with_alias_cos098"],
                                 "median_rel_distance_all_pairs": summ["median_rel_distance_all_pairs"], "roots": summ["roots"]})
            finally:
                mlf.end_run()
        print(json.dumps(summ), flush=True)
        print("top pairs:", top[:6], flush=True)


if __name__ == "__main__":
    main()
