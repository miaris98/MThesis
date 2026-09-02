#!/usr/bin/env bash
# ==============================================================================
# Re-apply this repo's shared_mode patch over the pip-installed easycarla package.
# ==============================================================================
# setup_vastai.sh does this once at install time, but the patch file changes
# whenever the vendored env is modified, and a plain `git pull` only updates the
# copy in this repo - not the installed package the training run actually imports.
# Run this after any pull that touches patches/easycarla_rl/carla_env.py.
# ==============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PATCH_FILE="$SCRIPT_DIR/patches/easycarla_rl/carla_env.py"

if [ ! -f "$PATCH_FILE" ]; then
    echo -e "${RED}[ERROR] Patch file not found: $PATCH_FILE${NC}"
    exit 1
fi

# carla_env.py does `import carla` at module level, so locating the installed
# package fails unless the CARLA .egg is on PYTHONPATH first.
export CARLA_ROOT=/workspace/carla
export PYTHONPATH="$(ls /workspace/carla/PythonAPI/carla/dist/carla-*-py3*.egg 2>/dev/null | tail -n 1):/workspace/carla/PythonAPI/carla:$PYTHONPATH"

TARGET=$(python -c "import easycarla.envs.carla_env as m, os; print(os.path.abspath(m.__file__))" 2>/dev/null || true)

if [ -z "$TARGET" ]; then
    echo -e "${RED}[ERROR] Could not locate the installed easycarla.envs.carla_env.${NC}"
    echo -e "${YELLOW}Check that the carla_py38 environment is active and that 'import carla' works:${NC}"
    echo -e "  conda activate carla_py38"
    echo -e "  python -c 'import carla; print(carla.__file__)'"
    exit 1
fi

cp "$PATCH_FILE" "$TARGET"
echo -e "${GREEN}✓ Applied shared_mode patch to $TARGET${NC}"

python - <<'PYCHECK'
import inspect
import easycarla.envs.carla_env as m
src = inspect.getsource(m.CarlaEnv.__init__)
missing = [f for f in ("shared_mode", "enable_lidar") if f not in src]
if missing:
    raise SystemExit(f"[ERROR] Patch applied but these flags are missing: {', '.join(missing)}")
print("✓ Verified: shared_mode and enable_lidar both present in the installed package.")
PYCHECK
