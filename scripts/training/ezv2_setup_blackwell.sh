#!/usr/bin/env bash
# EZ-V2 (upstream commit 12b9e77; code unchanged except env overrides for the Ray object store and CPU count) on Blackwell.
# The official stack (py3.8 / torch<=2.4.1 / ray 1.0) has no sm_120 kernels, so this ports the ENVIRONMENT only:
# py3.10, torch 2.7.1 cu128, ray 2.9.3, numpy 1.23.5 (gym 0.22 needs the pre-1.24 aliases),
# torchrl/tensordict matched to torch 2.7. Algorithm code/config = the box3 run (S-086).
set -ex
cd /workspace/EfficientZeroV2
git rev-parse HEAD
rm -rf /workspace/venv_ezv2
uv venv --python 3.10 /workspace/venv_ezv2
export VIRTUAL_ENV=/workspace/venv_ezv2 PATH=/workspace/venv_ezv2/bin:$PATH
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install torch==2.7.1 torchvision==0.22.1 "numpy==1.23.5" "gym[atari,accept-rom-license]==0.22.0" cython hydra-core \
  "protobuf==3.20.1" "wandb==0.14.2" "opencv-python<4.11" "kornia<0.8" "imageio[ffmpeg,pyav]==2.31.5" "ray==2.9.3" \
  "torchrl==0.8.1" "tensordict==0.8.3" colorednoise mlflow tensorboard "pillow<11" \
  --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install dm_control "git+https://github.com/denisyarats/dmc2gym.git" "numpy==1.23.5"  # ez/envs imports dmc2gym even for Atari
for d in ctree ctree_v2 ori_ctree; do (cd ez/mcts/$d && bash make.sh); done
# box3's exact yaml (atari.yaml with game=Breakout)
cp /workspace/atari_breakout.yaml ez/config/exp/atari_breakout.yaml
# 150 GB object store is hard-coded; make it an env override (default unchanged)
grep -q EZV2_OBJECT_STORE_GB ez/train.py || sed -i "s#object_store_memory=150 \* 1024 \* 1024 \* 1024 if config.env.image_based else 100 \* 1024 \* 1024 \* 1024#object_store_memory=int(float(os.environ.get(\"EZV2_OBJECT_STORE_GB\", 150 if config.env.image_based else 100)) * 1024 ** 3)#" ez/train.py
# cpu_count() sees every host CPU; vast containers cap pids (~2816), so Ray must be told the real quota
grep -q EZV2_NUM_CPUS ez/train.py || sed -i "s#num_cpus = multiprocessing.cpu_count()#num_cpus = int(os.environ.get(\"EZV2_NUM_CPUS\", multiprocessing.cpu_count()))#" ez/train.py
grep -n -E "EZV2_OBJECT_STORE_GB|EZV2_NUM_CPUS" ez/train.py
PYTHONPATH=. python -c "import torch,ray,gym,hydra,wandb,torchrl,numpy,ez.mcts,ez.agents,ez.worker; print(torch.__version__,torch.cuda.get_device_name(0),ray.__version__,gym.__version__,numpy.__version__,torchrl.__version__)"
echo EZV2_SETUP_DONE
