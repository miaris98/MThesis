# Struggle & Solutions Log
> **Policy**: Append-only. Each entry is timestamped and tagged. If a new fix contradicts an existing entry, it must reference the original by ID and the user is prompted for clarification before appending.

---

## [S-001] `conda activate` fails — `No such file or directory: /opt/conda/bin/activate`
**Date**: 2026-09-01  
**Environment**: Vast.ai `root@C.49520912`, `carla_py38` conda env  
**Symptom**: `source /opt/conda/bin/activate: No such file or directory`  
**Root Cause**: Miniconda was installed to `/workspace/miniconda` not `/opt/conda`.  
**Fix**: Use `conda activate carla_py38` directly (conda shell hook already initialized) or source from the correct prefix: `source /workspace/miniconda/etc/profile.d/conda.sh && conda activate carla_py38`.

---

## [S-002] Pretrained LAV checkpoint download fails
**Date**: 2026-09-01  
**Symptom**: `wget` 404 on `github.com/dotchen/LAV/releases/download/v0.1/lav_carla_pretrained.pth`  
**Root Cause**: LAV public release URL moved/removed.  
**Fix**: Use the CARLA TransFuser++ checkpoint at `/workspace/pretrained_carla/model_0030_0.pth` (already present on the Vast.ai image). The ResNet-34 backbone loader strips known prefixes and loads with `strict=False`.

---

## [S-003] `is_stalled` KeyError in camera_easycarla_env.py
**Date**: 2026-09-01  
**File**: `src/envs/camera_easycarla_env.py:288`  
**Symptom**: `KeyError: 'is_stalled'` immediately on first step, both workers crash.  
**Root Cause**: New `WorldOnRailsReward.compute_reward()` returned a plain dict without the `is_stalled` key that the environment expected. All other reward functions use `self._blank_info()` which always includes it.  
**Fix**:
1. Updated `src/envs/rewards/world_on_rails.py` to use `self._blank_info(..., is_stalled=is_stalled)`.
2. Hardened `camera_easycarla_env.py:288` to use `sub_info.get("is_stalled", False)` defensively.  
**Commit**: `6b543a5`

---

## [S-004] Missing `reward, sub_info = compute_reward(...)` line — `NameError: sub_info`
**Date**: 2026-09-01  
**File**: `src/envs/camera_easycarla_env.py:287`  
**Symptom**: Workers crash immediately on first step with NameError.  
**Root Cause**: When fixing S-003, the `reward, sub_info = self.reward_calc.compute_reward(...)` line was accidentally omitted, leaving `sub_info` unbound.  
**Fix**: Restored the compute_reward line at `camera_easycarla_env.py:287`.  
**Commit**: `3a6c03a`

---

## [S-005] `time-out of 120000ms` CARLA sensor spawn on port 2004
**Date**: 2026-09-01  
**Symptom**: `RuntimeError: time-out of 120000ms while waiting for the simulator ... localhost:2004` during `camera_sensor.py:43 world.spawn_actor(...)`.  
**Root Cause**: Previous crashed Python session held CARLA port 2004's RPC socket in `TIME_WAIT`.  
**Fix**:
```bash
killall -9 CarlaUE4-Linux-Shipping CarlaUE4.sh python python3 2>/dev/null || true
fuser -k 2000/tcp 2001/tcp 2002/tcp 2004/tcp 2005/tcp 2006/tcp 2>/dev/null || true
sleep 3
```

---

## [S-006] `wor` policy-arch causes infinite `Connecting to Carla server...` loop
**Date**: 2026-09-01  
**Symptom**: CARLA servers verified online but workers loop indefinitely on connection, never completing `reset()`.  
**Root Cause**: TBD — suspected leftover GPU VRAM from previous session combined with WoR policy init path.  
**Workaround**: Use `qwen100m` + `lav` backbone with `--reward-fn=wor`:
```bash
bash run_multi_carla_training.sh --fresh --reward-fn=wor 1 10000 qwen100m lav Town01
```
**Status**: OPEN

---

## [S-007] CARLA initialization taking >10 dots (server stuck on startup)
**Date**: 2026-09-01  
**Symptom**: `[CARLA #1/2 | Port 2000] Waiting for server initialization...........` (many more dots than normal 4).  
**Root Cause**: Previous `wor` hang left zombie `CarlaUE4-Linux-Shipping` processes consuming GPU VRAM.  
**Diagnosis**:
```bash
nvidia-smi
ss -tlnp | grep -E '2000|2004'
tmux capture-pane -t carla_server_0 -p | tail -n 20
```
**Fix**: Kill by PID from `nvidia-smi`, free ports, restart with single env (`1` instead of `2`).

---

## [S-008] Off-Road episodes dominate training (62% within first 10 steps)
**Date**: 2026-09-01  
**Analysis**: Mean steps to off-road = 12.0 steps. 58.1% of off-roads while steering hard left (steer < -0.2).  
**Proposed Fixes**: Spawn grace boundary, steering slew rate limiter, stronger lane centering, zero-centered actor head init.  
**Status**: OPEN (partially mitigated by WoR policy tighter steer variance)

---

## [S-009] `training_config.py` default values broke `test_config.py`
**Date**: 2026-09-01  
**Symptom**: `AssertionError: 'wor' != 'qwen100m'` in `tests/test_config.py::test_default_config`.  
**Root Cause**: Changed `--policy-arch` default to `wor` while adding WoR support.  
**Fix**: Restored `policy_arch` default to `qwen100m` and `reward_fn` default to `custom_1`. WoR still available via CLI flags.  
**Commit**: `3a6c03a`
