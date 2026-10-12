#!/usr/bin/env bash
# CARLA eval box without the HF checkpoint step (checkpoints are scp'd from E: when the HF token is not on the box yet): CARLA 0.9.15 + venv_carla +
# leaderboard via launch_b2d20.sh setup, then PROVISION_DONE in /workspace/setup_carla.log where queue_A9c.sh waits for it.
set -uo pipefail
cd /workspace && mkdir -p checkpoints logs_prov
( cd /workspace/MThesis && bash scripts/eval/launch_b2d20.sh setup > /workspace/logs_prov/setup_carla.log 2>&1 && \
  VIRTUAL_ENV=/workspace/venv_carla uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> /workspace/logs_prov/setup_carla.log 2>&1 && \
  VIRTUAL_ENV=/workspace/venv_carla uv pip install lmdb wandb pyyaml >> /workspace/logs_prov/setup_carla.log 2>&1 && \
  /workspace/venv_carla/bin/python -c "import torch,timm,carla,py_trees;x=torch.ones(2,device='cuda');print('venv_carla torch',torch.__version__,x.sum().item())" >> /workspace/logs_prov/setup_carla.log 2>&1 \
  && echo CARLA_DONE >> /workspace/logs_prov/status || echo CARLA_FAILED >> /workspace/logs_prov/status )
if grep -q CARLA_DONE logs_prov/status && ! grep -q CARLA_FAILED logs_prov/status; then
  cp logs_prov/setup_carla.log setup_carla.log && echo PROVISION_DONE >> setup_carla.log
fi
echo "PROVISION_END $(date -u)" >> /workspace/logs_prov/status
