#!/usr/bin/env bash
# S-115 eval-only CARLA box: CARLA 0.9.15 + venv_carla + leaderboard, and the checkpoints the repeat evals need from the
# HF relay (sha-checked). No dataset, no training env. Writes PROVISION_DONE where queue_A9c.sh waits for it.
set -uo pipefail
cd /workspace && mkdir -p checkpoints logs_prov
( cd /workspace/MThesis && bash scripts/eval/launch_b2d20.sh setup > /workspace/logs_prov/setup_carla.log 2>&1 && \
  VIRTUAL_ENV=/workspace/venv_carla uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> /workspace/logs_prov/setup_carla.log 2>&1 && \
  VIRTUAL_ENV=/workspace/venv_carla uv pip install lmdb wandb pyyaml >> /workspace/logs_prov/setup_carla.log 2>&1 && \
  /workspace/venv_carla/bin/python -c "import torch,timm,carla,py_trees;x=torch.ones(2,device='cuda');print('venv_carla torch',torch.__version__,x.sum().item())" >> /workspace/logs_prov/setup_carla.log 2>&1 \
  && echo CARLA_DONE >> /workspace/logs_prov/status || echo CARLA_FAILED >> /workspace/logs_prov/status ) &
( (/venv/main/bin/python -c "import huggingface_hub" 2>/dev/null || uv pip install --python /venv/main/bin/python -q huggingface_hub) && \
  /venv/main/bin/python - > /workspace/logs_prov/ckpts.log 2>&1 <<'PY' && echo CKPT_DONE >> /workspace/logs_prov/status || echo CKPT_FAILED >> /workspace/logs_prov/status
import hashlib, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=open("/dev/shm/hf_token").read().strip()); repo = api.whoami()["name"] + "/mthesis-relay"
want = {"e15_20260930/carla_armE_aug1_hires": "model_epoch_015.pth",
        "armJ_20260929/carla_armJ_ft_obst": "model_epoch_020.pth",
        "armJ2K_20261001/carla_armJ_ft_obst_s1": "model_epoch_020.pth",
        "armJ23_20261002/carla_armJ_ft_obst_s2": "model_epoch_020.pth",
        "armJ23_20261002/carla_armJ_ft_obst_s3": "model_epoch_020.pth"}
lfs = {x.path: x.lfs.sha256 for p in {k.split("/")[0] for k in want}
       for x in api.list_repo_tree(repo, path_in_repo=p, recursive=True) if getattr(x, "lfs", None)}
for sub, model in want.items():
    dest = "/workspace/checkpoints/" + sub.split("/")[1]; os.makedirs(dest, exist_ok=True)
    for f in ("frozen_backbone.pth", model, "run_config.json"):
        p = hf_hub_download(repo, f"{sub}/{f}", token=api.token, local_dir="/workspace/hf_dl")
        os.replace(p, f"{dest}/{f}")
        h = hashlib.sha256(open(f"{dest}/{f}", "rb").read()).hexdigest()
        ref = lfs.get(f"{sub}/{f}")
        assert ref is None or ref == h, f"sha mismatch {sub}/{f}"
        print(dest, f, os.path.getsize(f"{dest}/{f}"), h[:16], flush=True)
PY
) &
wait
grep -q CARLA_DONE logs_prov/status && grep -q CKPT_DONE logs_prov/status && cp logs_prov/setup_carla.log setup_carla.log && echo PROVISION_DONE >> setup_carla.log
echo "PROVISION_END $(date -u)" >> /workspace/logs_prov/status
