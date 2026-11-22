#!/usr/bin/env bash
# S-115 eval-only CARLA box: CARLA 0.9.15 + venv_carla + leaderboard, and the checkpoints the repeat evals need from the
# HF relay (sha-checked). No dataset, no training env. Writes PROVISION_DONE where queue_A9c.sh waits for it.
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
set -uo pipefail
cd $WORKSPACE && mkdir -p checkpoints logs_prov
( cd $WORKSPACE/MThesis && bash scripts/eval/launch_b2d20.sh setup > $WORKSPACE/logs_prov/setup_carla.log 2>&1 && \
  VIRTUAL_ENV=$WORKSPACE/venv_carla uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128 >> $WORKSPACE/logs_prov/setup_carla.log 2>&1 && \
  VIRTUAL_ENV=$WORKSPACE/venv_carla uv pip install lmdb wandb pyyaml >> $WORKSPACE/logs_prov/setup_carla.log 2>&1 && \
  $WORKSPACE/venv_carla/bin/python -c "import torch,timm,carla,py_trees;x=torch.ones(2,device='cuda');print('venv_carla torch',torch.__version__,x.sum().item())" >> $WORKSPACE/logs_prov/setup_carla.log 2>&1 \
  && echo CARLA_DONE >> $WORKSPACE/logs_prov/status || echo CARLA_FAILED >> $WORKSPACE/logs_prov/status ) &
( (/venv/main/bin/python -c "import huggingface_hub" 2>/dev/null || uv pip install --python /venv/main/bin/python -q huggingface_hub) && \
  /venv/main/bin/python - > $WORKSPACE/logs_prov/ckpts.log 2>&1 <<'PY' && echo CKPT_DONE >> $WORKSPACE/logs_prov/status || echo CKPT_FAILED >> $WORKSPACE/logs_prov/status
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
# original WoR (the thesis baseline, `WORL:` jobs): ship pcla_wor.tgz (built from Carla-utils/PCLA, 0.6 MB, E:/MThesis_EXP/live_20260929_ops_scripts/)
# with the code tgz; the weights come from the public PCLA HF repo. Without the tgz this step is skipped.
if [ -f $WORKSPACE/pcla_wor.tgz ]; then
  ( tar -xzf $WORKSPACE/pcla_wor.tgz -C $WORKSPACE && cd $WORKSPACE/PCLA &&     /venv/main/bin/python pcla_functions/download_weights.py --agents wor > $WORKSPACE/logs_prov/pcla.log 2>&1 &&     test -s pcla_agents/wor_pretrained/leaderboard_weights/main_model_10.th && echo PCLA_DONE >> $WORKSPACE/logs_prov/status ) || echo PCLA_FAILED >> $WORKSPACE/logs_prov/status
else echo PCLA_SKIPPED >> $WORKSPACE/logs_prov/status; fi
if grep -q CARLA_DONE logs_prov/status && grep -q CKPT_DONE logs_prov/status && ! grep -q PCLA_FAILED logs_prov/status; then
  cp logs_prov/setup_carla.log setup_carla.log && echo PROVISION_DONE >> setup_carla.log
fi
echo "PROVISION_END $(date -u)" >> $WORKSPACE/logs_prov/status
