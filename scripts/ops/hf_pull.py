import sys, os
from huggingface_hub import HfApi, snapshot_download
WORKSPACE = os.environ.get("WORKSPACE", "/workspace").rstrip("/")   # the box workspace (src/config/paths.py: WORKSPACE)
sub = sys.argv[1]
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
p = snapshot_download(repo, allow_patterns=[sub + "/*"], token=api.token, local_dir=f"{WORKSPACE}/hf_dl")
src = os.path.join(f"{WORKSPACE}/hf_dl", sub)
for d in os.listdir(src):
    os.system(f"mkdir -p {WORKSPACE}/checkpoints/{d} && cp -f {src}/{d}/* {WORKSPACE}/checkpoints/{d}/")
    print(d, os.listdir(f"{WORKSPACE}/checkpoints/{d}"))
