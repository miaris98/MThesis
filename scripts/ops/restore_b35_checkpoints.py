"""S-115 box N: restore the B35 30k checkpoints (env30000_upd28004, the B35 metric point) + their stored 10-episode
eval JSONs from the HF relay into the run folders, sha256-checked against HF's LFS records. `--dry` only lists.
Also times one 200 MB upload to HF (box uplink check)."""
import hashlib, os, re, sys, time
from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
ROOT = "/workspace/MThesis_atari/results/100k_benchmark/S058_ezv2_match"
WANT = ("checkpoint_env30000_upd28004.pt", "eval_env30000_upd28004.json")
RUN = re.compile(r"(S058[hi]_(gtrxl|resnet)_(?:30k|100k)_s\d+_uniform)")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


pick = {}  # (run, file) -> repo entry; first prefix wins
for prefix in ("b35_20261001", "b35_20261002_L", "b35_20261002_M", "b35_100k_env30000"):
    for x in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True):
        f = x.path.rsplit("/", 1)[-1]
        m = RUN.search(x.path)
        if hasattr(x, "size") and m and f in WANT:
            pick.setdefault((m.group(1), f), x)
runs = sorted({r for r, _ in pick})
print(len(runs), "runs:", " ".join(runs))
print("missing:", [(r, f) for r in runs for f in WANT if (r, f) not in pick])
if "--dry" in sys.argv:
    raise SystemExit(0)
bad = 0
for (run, f), x in sorted(pick.items()):
    dest = f"{ROOT}/{run}/checkpoints/{f}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    p = hf_hub_download(repo, x.path, token=api.token, local_dir="/workspace/hf_dl")
    os.replace(p, dest)
    ok = x.lfs is None or sha(dest) == x.lfs.sha256
    bad += not ok
    print("OK " if ok else "BAD", x.path, os.path.getsize(dest), flush=True)
print(f"RESTORE {'DONE' if bad == 0 else 'FAILED'} files={len(pick)} bad={bad}", flush=True)

os.makedirs("/workspace/hftest", exist_ok=True)
with open("/workspace/hftest/t.bin", "wb") as fh:
    fh.write(os.urandom(200_000_000))
t = time.time()
api.upload_file(path_or_fileobj="/workspace/hftest/t.bin", path_in_repo="speedtest/P.bin", repo_id=repo)
dt = time.time() - t
api.delete_file("speedtest/P.bin", repo_id=repo)
print(f"UPLINK to HF: 200 MB in {dt:.0f} s = {200 / dt:.1f} MB/s", flush=True)
