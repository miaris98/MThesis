"""Policy sharpness on two state sets (Qwen seed-1 replay and champion seed-2 replay):
  - entropy of the STORED search policy targets (pi) in each replay's second half (what the policy head is trained on)
  - each final model's prior entropy and P(FIRE) on both state sets, and the Qwen mixers' output/input ratio
  - the entropy a champion-sharp prior would have after the 0.05 target smoothing: H(0.95 p + 0.0125)
Reuses the state loader and model builder of mixer_diag.py."""
import os, sys, math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mixer_diag as md  # noqa: E402

REPLAYS = {"qwen_s1_replay": md.REPLAY, "champ_s2_replay": os.path.join(HERE, "champ_s2_replay.npz")}
FINAL = md.FINAL
MODELS = {
    "qwen_s1_u120k": FINAL + r"\b100k_qnowd_s1\checkpoints\checkpoint_env100000_upd120000.pt",
    "qwen_s2_u120k": FINAL + r"\b100k_qnowd_s2\checkpoints\checkpoint_env100000_upd120000.pt",
    "qwen_s2_u98k": os.path.join(HERE, "ckpts", "s2_env100000_upd098004.pt"),
    "champ_s1": md.CHAMP + r"\s1\checkpoint_latest.pt",
    "champ_s2": md.CHAMP + r"\s2\checkpoint_env100000_upd120000.pt",
    "champ_s3": md.CHAMP + r"\s3\checkpoint_latest.pt",
    "champ_s4": md.CHAMP + r"\s4\checkpoint_env100000_upd120000.pt",
}
H = lambda p: -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def stored_targets(path):
    pi, ep_start = md.npz_memmap(path, "pi.npy"), md.npz_memmap(path, "ep_start.npy")
    T = ep_start.shape[0]
    x = np.asarray(pi[T // 2:]).reshape(-1, pi.shape[-1]).astype(np.float64)
    x = x[x.sum(-1) > 0.5]
    return float(H(x).mean()), float(x[:, 1].mean())


if __name__ == "__main__":
    for name, path in REPLAYS.items():
        h, f = stored_targets(path)
        print(f"{name}: stored search targets, 2nd half: entropy {h:.3f} nats, mean pi(FIRE) {f:.3f}")
    built = {k: md.build(p)[0] for k, p in MODELS.items() if os.path.exists(p)}
    for name, path in REPLAYS.items():
        md.REPLAY = path
        obs, act = md.states()
        print(f"\n== states from {name}")
        for k, m in built.items():
            with torch.no_grad():
                s = m.representation(obs)
                _, p = m.prediction(s)
                prob = p.softmax(-1).numpy().astype(np.float64)
                line = f"{k:14s} prior H {H(prob).mean():.3f}  P(FIRE) {prob[:, 1].mean():.3f}"
                if k.startswith("champ"):
                    line += f"  | after 0.05 smoothing H {H(0.95 * prob + 0.0125).mean():.3f}"
                if m.trunk == "qwen":
                    _, ratio = md.mixer_stats(m.repr_mixer, m.repr_res(m.down(obs)))
                    line += f"  | repr mixer ratio {ratio:.2f}"
            print(line, flush=True)
