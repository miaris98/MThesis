# Restore the GTrXL 80k seed-0 run (2026-09-29 final save) from the HF relay's mthesis-relay/s0_20260929_final/
# into results/100k_benchmark/S058_ezv2_match/, sha256-checked against HF's LFS record. --dry lists only.
import hashlib, os, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
PREFIX = "s0_20260929_final/S058h_gtrxl_100k_s0_uniform"
DEST = "/workspace/MThesis/results/100k_benchmark/S058_ezv2_match/S058h_gtrxl_100k_s0_uniform/checkpoints"
os.makedirs(DEST, exist_ok=True)
files = [t for t in api.list_repo_tree(repo, path_in_repo=PREFIX, recursive=True)]
print(f"{len(files)} files under {PREFIX}:")
for f in files:
    print(" ", f.path, getattr(f, "size", "?"))
if "--dry" in sys.argv:
    sys.exit(0)
for f in files:
    name = f.path.rsplit("/", 1)[-1]
    p = hf_hub_download(repo, f.path, token=api.token, local_dir="/workspace/hf_dl")
    dest = os.path.join(DEST, name)
    import shutil; shutil.copy(p, dest)
    sha = getattr(getattr(f, "lfs", None), "sha256", None)
    if sha:
        ok = hashlib.sha256(open(dest, "rb").read()).hexdigest() == sha
        print(name, "OK" if ok else "SHA MISMATCH")
    else:
        print(name, "copied (no lfs sha to check)")
