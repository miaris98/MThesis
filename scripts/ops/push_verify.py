"""S-113: push a box's result folder to the HF relay (no replay buffers) and sha256-check every file against HF.
usage: push_verify.py LOCAL_ROOT REPO_SUBDIR"""
import glob, hashlib, os, sys
from huggingface_hub import HfApi

root, sub = sys.argv[1], sys.argv[2]
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
skip = lambda p: "replay" in p or ".tmp" in p
for attempt in range(5):
    try:
        api.upload_folder(folder_path=root, path_in_repo=sub, repo_id=repo, ignore_patterns=["*replay*", "*.tmp*"])
        break
    except Exception as ex:
        print("retry upload", attempt, repr(ex)[:150], flush=True)
rem = {x.path[len(sub.rstrip("/")) + 1:]: (x.lfs.sha256 if x.lfs else None, x.size)  # works for nested subdirs
       for x in api.list_repo_tree(repo, path_in_repo=sub, recursive=True) if hasattr(x, "size")}
n = bad = 0
for p in glob.glob(root + "/**/*", recursive=True):
    if not os.path.isfile(p) or skip(p):
        continue
    k = os.path.relpath(p, root); n += 1; s, sz = rem.get(k, (None, None))
    real = os.path.realpath(p)
    ok = sz == os.path.getsize(real) and (s is None or s == hashlib.sha256(open(real, "rb").read()).hexdigest())
    if not ok:
        bad += 1; print("BAD", k, flush=True)
print(f"PUSH_VERIFY {sub}: files {n} bad {bad}", flush=True)
