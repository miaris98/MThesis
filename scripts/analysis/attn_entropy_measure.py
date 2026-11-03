"""Gate attention check (2026-11-03): rebuild each checkpoint's network from its saved args, load it, and measure the conv trunk's
normalised attention entropy on states sampled from that run's replay (last 4 frames, /255, as the trainer's stacks())."""
import sys
import numpy as np
import torch

from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport


def ga(a, k, d=None):
    return a.get(k, d) if isinstance(a, dict) else getattr(a, k, d)


def measure(ck_path, replay_path, n=256, seed=0):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    m = EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk=ga(a, "trunk"), norm=ga(a, "norm", "batch"),
                  state_hw=int(np.ceil(ga(a, "frame_size", 84) / 16)), mixer_out_init_std=ga(a, "mixer_out_init_std", 0.0),
                  mixer_bg_init=ga(a, "mixer_bg_init", 0.0), simnorm=ga(a, "simnorm", 0), mixer_pos=ga(a, "mixer_pos", "rope"),
                  mixer_size=ga(a, "mixer_size", "small"), mixer_qk_norm=ga(a, "mixer_qk_norm", False),
                  mixer_where=ga(a, "mixer_where", "pred"), mixer_no_pos=ga(a, "mixer_no_pos", False))
    m.load_state_dict(ck["model"], strict=True)
    m.eval()
    r = np.load(replay_path)
    frames, T = r["frames"], r["frames"].shape[0]
    rng = np.random.default_rng(seed)
    t = rng.integers(3, T, size=n)
    e = rng.integers(0, frames.shape[1], size=n)
    obs = torch.from_numpy(np.stack([frames[t - o, e] for o in (3, 2, 1, 0)], 1).reshape(n, 12, 96, 96)).float().div_(255.0)
    layers, pols = gate_attention(m, obs)
    return ck["updates"], ck["env_steps"], layers, pols


@torch.no_grad()
def gate_attention(model, obs: torch.Tensor, chunk: int = 64):
    """Mean normalised attention entropy per layer and for the policy query, over RAW observation stacks (float in [0, 1], (n, 12, H, W)).
    The trainer's in-run log and the offline gate both call this, so they measure the same thing. Eval mode while measuring (batch-norm
    running stats untouched); the training flag is restored."""
    was_training = model.training
    model.eval()
    layers, pols = [], []
    for i in range(0, len(obs), chunk):
        s, _, _ = model.initial_inference(obs[i:i + chunk])
        rep = model.pred_mixer.attention_entropy(s.float())
        layers.append(rep["per_layer"])
        pols.append(rep["policy_row"])
    model.train(was_training)
    return np.mean(layers, 0), np.mean(pols, 0)


if __name__ == "__main__":
    args = sys.argv[1:]
    for ck_path, rp in zip(args[0::2], args[1::2]):
        upd, env, L, P = measure(ck_path, rp)
        per = " / ".join("%.2f" % x for x in L)
        print("%s env %s upd %s: per layer %s | policy %.2f" % (ck_path, env, upd, per, float(np.mean(P))), flush=True)
