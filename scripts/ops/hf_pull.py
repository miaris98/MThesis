import sys, os
from huggingface_hub import HfApi, snapshot_download
sub = sys.argv[1]
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
p = snapshot_download(repo, allow_patterns=[sub + "/*"], token=api.token, local_dir="/workspace/hf_dl")
src = os.path.join("/workspace/hf_dl", sub)
for d in os.listdir(src):
    os.system(f"mkdir -p /workspace/checkpoints/{d} && cp -f {src}/{d}/* /workspace/checkpoints/{d}/")
    print(d, os.listdir(f"/workspace/checkpoints/{d}"))
