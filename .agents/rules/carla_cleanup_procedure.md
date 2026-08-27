# CARLA Process Cleanup & Socket Reset Rule

## Mandatory Cleanup Before Starting / Restarting CARLA Runs
Whenever stopping, resetting, or launching a new CARLA training or evaluation session, always ensure all stale CARLA server processes, child simulator threads, and locked RPC ports are completely terminated first.

### Why this is necessary:
- When a CARLA training session is interrupted or ends, Unreal Engine 4 and the CARLA server often remain frozen in synchronous mode or hold onto TCP ports (2000, 2001, 2002, 2004, 2005, 2006).
- Attempting to connect a new Python client to a stale or frozen CARLA server will cause `RuntimeError: time-out of 120000ms while waiting for the simulator` or sensor unsubscribe warnings (`Actor 100-104 sensor wasn't listening`).

### Mandatory 1-Line Clean Reset Command:
```bash
pkill -9 -f CarlaUE4; pkill -9 -f train_rl_agent; fuser -k -9 2000/tcp 2001/tcp 2002/tcp 2004/tcp 2005/tcp 2006/tcp 2>/dev/null; sleep 2
```
Always execute or recommend this cleanup before starting fresh training or evaluation runs.
