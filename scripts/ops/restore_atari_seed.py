"""Restore an Atari seed's resume state (checkpoint_latest.pt + replay_latest.npz.gz + small eval/log files) from the HF relay.
usage: restore_atari_seed.py SEED NAME [--dry]     NAME = the live_20261010/<NAME> folder the seed was backed up under
  (2026-10-10 names: seed1=R7900X, seed2=D-xeon, seed3=R5950X, seed4=R3600-s4, seed0=C1-s0)
Token from /dev/shm/hf_token (put there by scripts/ops/push_hf_token_to_boxes.sh). Every file is sha256-checked against HF's LFS record.
Periodic checkpoint_env*.pt files are skipped (not needed to resume)."""
import hashlib, os, shutil, sys
from huggingface_hub import HfApi, hf_hub_download

seed, name = sys.argv[1], sys.argv[2]
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
rel = f"MThesis/results/100k_benchmark/S058_ezv2_match/S058h_gtrxl_100k_s{seed}_uniform"
prefix = f"live_20261010/{name}/{rel}"
dest_root = f"/workspace/{rel}"
files = [t for t in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True) if hasattr(t, "size")]
keep = [f for f in files if not os.path.basename(f.path).startswith("checkpoint_env")]
print(f"{len(files)} files under {prefix}; restoring {len(keep)}:")
for f in keep:
    print(" ", f.path[len(prefix) + 1:], f.size)
if "--dry" in sys.argv:
    sys.exit(0)
need = {"checkpoint_latest.pt", "replay_latest.npz.gz"}
have = {os.path.basename(f.path) for f in keep}
if not need <= have:
    sys.exit(f"MISSING {need - have} on HF; refusing to resume without the checkpoint+replay pair")
bad = 0
for f in keep:
    sub = f.path[len(prefix) + 1:]
    p = hf_hub_download(repo, f.path, token=api.token, local_dir="/workspace/hf_dl")
    dest = os.path.join(dest_root, sub); os.makedirs(os.path.dirname(dest), exist_ok=True); shutil.copy(p, dest)
    sha = getattr(getattr(f, "lfs", None), "sha256", None)
    ok = (sha is None) or hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha
    bad += not ok
    print(sub, "OK" if ok else "SHA MISMATCH", flush=True)
sys.exit(1 if bad else 0)
