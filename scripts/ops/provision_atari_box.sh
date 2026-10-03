#!/usr/bin/env bash
# Atari box (vast.ai pytorch image, /venv/main): HF reachability FIRST (S-115 box N had censored DNS), then deps, the diagnostics state
# file and the checkpoints. Run on the box with /workspace/MThesis already unpacked (tar of `git ls-files` over ssh) and the HF token
# already in /dev/shm/hf_token (sent over stdin, never printed).
# usage: provision_atari_box.sh [restore]     (restore = also pull the 16 B35 env30000 checkpoints, ~1.7 GB)
set -uo pipefail
cd /workspace
ip=$(getent ahosts huggingface.co | head -1 | cut -d' ' -f1)
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 https://huggingface.co || true)
echo "huggingface.co -> $ip, HTTP $code"
[ "$code" = 200 ] || { echo "HF NOT REACHABLE: tell the user to destroy this box (S-115 box N)"; exit 1; }
nproc; free -g | sed -n 2p; df -h /workspace | tail -1; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv pip install --python /venv/main/bin/python "gymnasium[atari]" ale-py opencv-python-headless mlflow tensorboard huggingface_hub > deps.log 2>&1 \
  && echo DEPS_OK || { echo DEPS_FAILED; tail -3 deps.log; exit 1; }
/venv/main/bin/python - <<'PY'
import hashlib
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
x = [t for t in api.list_repo_tree(repo, path_in_repo="diag") if t.path.endswith("b15_states_512.npy")][0]
p = hf_hub_download(repo, x.path, token=api.token, local_dir="/workspace/hf_dl")
import shutil; shutil.copy(p, "/workspace/b15_states_512.npy")
ok = hashlib.sha256(open("/workspace/b15_states_512.npy", "rb").read()).hexdigest() == x.lfs.sha256
print("states file", "OK" if ok else "SHA MISMATCH")
PY
if [ "${1:-}" = restore ]; then /venv/main/bin/python /workspace/MThesis/scripts/ops/restore_b35_checkpoints.py; fi
/venv/main/bin/python -c "import torch, gymnasium, ale_py; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
