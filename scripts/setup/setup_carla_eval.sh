#!/usr/bin/env bash
# Prepares this instance for closed-loop CARLA evaluation.
#   1. NVIDIA userspace graphics driver (exact kernel-module version, no kernel module)
#   2. CARLA 0.9.15 server + AdditionalMaps (Town06/07/13/15 - NOT in the base package)
#   3. A Python 3.10 venv, because the carla wheel has no cp312 ABI tag and /venv/main is 3.12
# Follows carla_vulkan_driver_troubleshooting_guide.md sections 4.2 and setup_vastai.sh step 2.
set -x

echo "===== [1/7] apt prerequisites ====="
apt-get update -qq
apt-get install -y -qq kmod aria2 wget libtiff5-dev 2>/dev/null || apt-get install -y -qq kmod aria2 wget

echo "===== [2/7] NVIDIA userspace driver (matching kernel module exactly) ====="
DRIVER_VERSION=$(cat /proc/driver/nvidia/version | grep -oP '(?<=Module  )[0-9.]+' | head -1)
echo "Kernel driver version: ${DRIVER_VERSION}"
cd /workspace
if [ ! -f "NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" ]; then
  wget -q --show-progress "https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" \
    || wget -q --show-progress "https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
fi
# --no-kernel-module is mandatory: the kernel module belongs to the vast.ai host.
sh "/workspace/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" \
  --no-kernel-module --silent --install-libglvnd \
  --no-nvidia-modprobe --no-rebuild-initramfs

echo "===== [3/7] Verify Vulkan ====="
ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* || echo "MISSING libGLX_nvidia"
vulkaninfo --summary 2>&1 | head -20

echo "===== [4/7] CARLA 0.9.15 ====="
CARLA_DIR=/workspace/carla
if [ -f "$CARLA_DIR/CarlaUE4.sh" ]; then
  echo "CARLA already present, skipping."
else
  mkdir -p "$CARLA_DIR"
  URL_S3="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz"
  aria2c -x 16 -s 16 -k 1M --check-certificate=false "$URL_S3" -d /workspace -o CARLA_0.9.15.tar.gz \
    || wget -c "$URL_S3" -O /workspace/CARLA_0.9.15.tar.gz
  tar -xf /workspace/CARLA_0.9.15.tar.gz -C "$CARLA_DIR"
  rm -f /workspace/CARLA_0.9.15.tar.gz
fi

echo "===== [5/7] CARLA AdditionalMaps (Town06/07/13/15) ====="
# Base package above only ships Town01-05 + Town10HD. bench2drive220.xml routes reference
# Town06/07/13/15 too - without this, every route on those towns burns its full per-route
# timeout on "RuntimeError: Map 'TownXX' not found" and comes back with zero data (found
# 2026-09-21: 16/38 Bench2Drive routes silently failed this way on a freshly provisioned box).
# The official S3/backblaze CDN 403s for this asset (unlike the base package above) - use the
# downloads.carlasim.com mirror instead, confirmed working.
if [ -f "$CARLA_DIR/CarlaUE4/Content/Carla/Maps/Town06.umap" ]; then
  echo "AdditionalMaps already present, skipping."
else
  URL_MAPS="https://downloads.carlasim.com/Linux/AdditionalMaps_0.9.15.tar.gz"
  aria2c -x 16 -s 16 -k 1M --check-certificate=false "$URL_MAPS" -d /workspace -o AdditionalMaps_0.9.15.tar.gz \
    || wget -c "$URL_MAPS" -O /workspace/AdditionalMaps_0.9.15.tar.gz
  tar -xzf /workspace/AdditionalMaps_0.9.15.tar.gz -C "$CARLA_DIR"
  rm -f /workspace/AdditionalMaps_0.9.15.tar.gz
fi

id carlauser >/dev/null 2>&1 || useradd -m -s /bin/bash carlauser
chown -R carlauser:carlauser "$CARLA_DIR"
ls "$CARLA_DIR"
echo "--- installed maps ---"
ls "$CARLA_DIR/CarlaUE4/Content/Carla/Maps/" | grep -iE 'town' || echo "MISSING maps directory"
echo "--- shipped PythonAPI distributions ---"
ls "$CARLA_DIR/PythonAPI/carla/dist/" 2>&1

echo "===== [6/7] Python 3.10 venv for the carla client ====="
cd /workspace/MThesis
uv venv --python 3.10 /workspace/venv_carla
/workspace/venv_carla/bin/python -V
# Prefer the wheel shipped with the server (guaranteed protocol match); fall back to PyPI.
CARLA_WHL=$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-cp310-*.whl 2>/dev/null | head -1)
if [ -n "$CARLA_WHL" ]; then
  VIRTUAL_ENV=/workspace/venv_carla uv pip install "$CARLA_WHL"
else
  VIRTUAL_ENV=/workspace/venv_carla uv pip install carla==0.9.15
fi
VIRTUAL_ENV=/workspace/venv_carla uv pip install \
  "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
VIRTUAL_ENV=/workspace/venv_carla uv pip install numpy opencv-python pillow pandas pytest

echo "===== [7/7] Verify client import ====="
/workspace/venv_carla/bin/python -c "import carla, torch; print('carla', carla.__file__); print('torch', torch.__version__, torch.cuda.is_available())"

echo "SETUP_CARLA_EVAL_DONE"
