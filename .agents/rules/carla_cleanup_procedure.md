---
trigger: always_on
description: Mandatory CARLA server process cleanup and RPC port reset before starting or restarting CARLA runs.
---

# CARLA Process Cleanup & Socket Reset Rule

## Mandatory Cleanup Before Starting / Restarting CARLA Runs
Whenever stopping, resetting, or launching a new CARLA training or evaluation session, always ensure all stale CARLA server processes, child simulator threads, and locked RPC ports are completely terminated first.

### Why this is necessary:
- When a CARLA training or evaluation session is interrupted or ends, Unreal Engine 4 and the CARLA server often remain frozen in synchronous mode or hold onto TCP ports (2000, 2001, 2002, 2004, 2005, 2006, etc., and 2–6 GB VRAM per server).
- Attempting to connect a new Python client to a stale or frozen CARLA server will cause `RuntimeError: time-out of 120000ms while waiting for the simulator`, sensor unsubscribe warnings (`Actor 100-104 sensor wasn't listening`), or CUDA OOM when launching a new wave of lanes.

### Safe Cleanup Practice:
- Prefer `clear_carla_ports` / `kill_tree` from `scripts/lib/common.sh` or killing orphaned `-carla-rpc-port=` servers whose ports are not in the active lane list.
- For a full reset on a dedicated CARLA box (executed from a script file so `pkill -f` does not self-kill the SSH shell):
```bash
pkill -9 -f CarlaUE4; pkill -9 -f train_rl_agent; fuser -k -9 2000/tcp 2001/tcp 2002/tcp 2004/tcp 2005/tcp 2006/tcp 2>/dev/null; sleep 2
```
Always execute or recommend this cleanup before starting fresh training or evaluation runs.
