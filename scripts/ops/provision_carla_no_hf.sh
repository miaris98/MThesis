#!/usr/bin/env bash
# CARLA eval box without the HF checkpoint step (checkpoints are scp'd from E: when the HF token is not on the box yet): CARLA 0.9.15 + venv_carla +
# leaderboard via launch_b2d20.sh setup, then PROVISION_DONE in /workspace/setup_carla.log where queue_A9c.sh waits for it.
set -uo pipefail
cd /workspace && mkdir -p checkpoints logs_prov
# 2026-10-28 (box 58066117, driver 555.58.02): the userspace driver install reruns ldconfig, which then put the image's CUDA forward-compat
# libcuda (/usr/local/cuda-12.6/compat, 560.x) ahead of the host's: CUDA error 803 in every Python. Compat libs are disabled after the setup.
# The cu128 wheels also need driver >= 570; older drivers get cu124 (torch 2.6.0).
DRV=$(head -1 /proc/driver/nvidia/version | grep -oE '[0-9]{3}\.[0-9]+' | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -ge 570 ]; then TORCH_IDX=cu128; TORCH_PKGS="torch torchvision"; else TORCH_IDX=cu124; TORCH_PKGS="torch==2.6.0 torchvision==0.21.0"; fi
( cd /workspace/MThesis && bash scripts/eval/launch_b2d20.sh setup > /workspace/logs_prov/setup_carla.log 2>&1 && \
  { for f in $(grep -l "cuda.*compat" /etc/ld.so.conf.d/* 2>/dev/null); do mv "$f" "/root/$(basename "$f").disabled_compat"; done; ldconfig; } && \
  VIRTUAL_ENV=/workspace/venv_carla uv pip install --reinstall $TORCH_PKGS --index-url https://download.pytorch.org/whl/$TORCH_IDX >> /workspace/logs_prov/setup_carla.log 2>&1 && \
  VIRTUAL_ENV=/workspace/venv_carla uv pip install lmdb wandb pyyaml >> /workspace/logs_prov/setup_carla.log 2>&1 && \
  /workspace/venv_carla/bin/python -c "import torch,timm,carla,py_trees;x=torch.ones(2,device='cuda');print('venv_carla torch',torch.__version__,x.sum().item())" >> /workspace/logs_prov/setup_carla.log 2>&1 \
  && echo CARLA_DONE >> /workspace/logs_prov/status || echo CARLA_FAILED >> /workspace/logs_prov/status )
if grep -q CARLA_DONE logs_prov/status && ! grep -q CARLA_FAILED logs_prov/status; then
  cp logs_prov/setup_carla.log setup_carla.log && echo PROVISION_DONE >> setup_carla.log
fi
echo "PROVISION_END $(date -u)" >> /workspace/logs_prov/status
