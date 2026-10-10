"""Push a transit folder to the HF relay under SUB and sha-check it. usage: hf_push_dir.py DIR SUB ; token from env HF_TOKEN. Prints one PUSH_VERIFY line."""
import glob, hashlib, os, sys
from huggingface_hub import HfApi

root, sub = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"]); repo = api.whoami()["name"] + "/mthesis-relay"
for attempt in range(4):
    try:
        api.upload_folder(folder_path=root, path_in_repo=sub, repo_id=repo); break
    except Exception as ex:
        print("retry", attempt, repr(ex)[:120], flush=True)
rem = {x.path[len(sub) + 1:]: (x.lfs.sha256 if x.lfs else None, x.size) for x in api.list_repo_tree(repo, path_in_repo=sub, recursive=True) if hasattr(x, "size")}
n = bad = 0
for p in glob.glob(root + "/**/*", recursive=True):
    if not os.path.isfile(p):
        continue
    k = os.path.relpath(p, root).replace("\\", "/"); n += 1; s, sz = rem.get(k, (None, None))
    if sz != os.path.getsize(p) or (s is not None and s != hashlib.sha256(open(p, "rb").read()).hexdigest()):
        bad += 1
print(f"PUSH_VERIFY {sub}: files {n} bad {bad}", flush=True)
