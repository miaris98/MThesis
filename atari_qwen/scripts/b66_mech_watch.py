"""Box side, CPU only (TODO B66 step 1): mechanism diagnostics for every 10k checkpoint of b66_fix_s1 and b66_sgd_s1 while they train.

On the SAME 192 reference states as the 2026-10-26 diagnosis (qnowd seed-1 replay, second half; ref_states_qnowd_s1.npz), via mixer_diag.diag():
mixer output / state ratio (representation and one dynamics step), attention entropy and peak per block, alpha gates, policy prior entropy,
P(FIRE), value. Also the entropy of each run's own acting-time search policies over its latest 10k env steps (replay_latest.npz `pi`).
Appends one JSON line per checkpoint to /workspace/b66/mech/mech.jsonl and prints a compact line. Exits after the training done marker
once every checkpoint is processed.
usage: OMP_NUM_THREADS=4 nice -n 19 python b66_mech_watch.py"""
import glob, json, os, re, sys, time, traceback
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mixer_diag as md  # noqa: E402

torch.set_num_threads(4)
OUT = "/workspace/b66/out"
# comma lists from the environment (B67 added runs while the watcher was live); it exits once every done marker exists and nothing is left
LABELS = tuple(os.environ.get("MECH_LABELS", "b66_fix_s1,b66_sgd_s1").split(","))
DONES = tuple(os.environ.get("MECH_DONE", "/workspace/b66/logs/done_s1.marker").split(","))
RES = "/workspace/b66/mech/mech.jsonl"
H = lambda p: -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def recent_search_entropy(label, rows=2500):
    """Entropy of the stored acting-time search policies over the last `rows` replay rows (4 envs each = 10k env steps)."""
    p = f"{OUT}/{label}/checkpoints/replay_latest.npz"
    if not os.path.exists(p):
        return None
    with np.load(p) as z:  # reads only the small `pi` member (~1.6 MB); md.npz_memmap needs a private numpy 1.x helper
        pi = z["pi"]
    x = np.asarray(pi[-rows:]).reshape(-1, pi.shape[-1]).astype(np.float64)
    x = x[x.sum(-1) > 0.5]
    return {"search_pi_entropy_recent": float(H(x).mean()), "search_pi_fire_recent": float(x[:, 1].mean()), "replay_rows": int(pi.shape[0])}


def main():
    ref = np.load(os.path.join(HERE, "ref_states_qnowd_s1.npz"))
    obs = torch.from_numpy(ref["obs"]).float().div_(255.0)
    act = torch.from_numpy(ref["act"]).long()
    os.makedirs(os.path.dirname(RES), exist_ok=True)
    done = set()
    if os.path.exists(RES):
        done = {json.loads(l)["ckpt"] for l in open(RES)}
    pat = re.compile(r"checkpoint_env(\d+)_upd(\d+)\.pt$")
    while True:
        finished = all(os.path.exists(d) for d in DONES)  # checked before the scan: after the markers every write is complete
        todo = []
        for label in LABELS:
            for p in sorted(glob.glob(f"{OUT}/{label}/checkpoints/checkpoint_env*_upd*.pt")):
                if pat.search(p) and p not in done and (finished or time.time() - os.path.getmtime(p) > 60):
                    todo.append((label, p))
        for label, p in todo:
            try:
                r = md.diag(p, obs, act)
                r["run"] = label
                try:
                    r.update(recent_search_entropy(label) or {})
                except Exception as e:  # the replay is replaced atomically; a failure here only skips this extra
                    r["search_pi_error"] = repr(e)
            except Exception:
                print(f"{time.strftime('%H:%M')} {label} {os.path.basename(p)} FAILED, retry next pass\n{traceback.format_exc()}", flush=True)
                continue
            with open(RES, "a") as f:
                f.write(json.dumps(r) + "\n")
            done.add(p)
            rb, db = r["repr_blocks"], r["dyn_blocks"]
            print(f"{time.strftime('%H:%M')} {label} env {r['env_steps']} upd {r['updates']} | repr ratio {r['repr_ratio']:.3f} dyn ratio {r['dyn_ratio']:.3f} "
                  f"| repr ent {rb[0]['entropy']:.3f}/{rb[1]['entropy']:.3f} peak {rb[0]['peak']:.2f}/{rb[1]['peak']:.2f} "
                  f"| logit max {max(b['logit_absmax'] for b in rb + db):.1f} | alpha a/f {rb[0]['alpha_attn']:.2f}/{rb[0]['alpha_ffn']:.2f} "
                  f"| prior H {r['policy_entropy']:.3f} P(FIRE) {r['p_fire']:.3f} | search H recent {r.get('search_pi_entropy_recent', float('nan')):.3f} "
                  f"| V {r['value_mean']:.2f}", flush=True)
        if finished and all(p in done for _, p in todo):
            print(f"{time.strftime('%H:%M')} MECH DONE", flush=True)
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
