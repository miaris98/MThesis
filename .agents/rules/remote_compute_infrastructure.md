---
trigger: always_on
description: Box-agnostic remote compute infrastructure (.remote, remote_config.py, machine.py), remote launch verification, SSH/process hygiene, and queue pitfalls.
---

# Remote Compute Infrastructure & Launch Hygiene Rules

## 1. Box-Agnostic Script Requirement
All scripts in `scripts/setup/` and `scratch/` that touch a remote compute box **MUST** use the central config loader. **Never hardcode SSH host, port, user, workspace path, venv path, or CPU core ranges** anywhere in script source files.

## 2. The Config Layer
- **Config file** (git-ignored, one per box): `.remote` at the repository root (never read directly — use `remote_config.py` or `python scripts/setup/vast_fleet.py select <idx_or_id>`).
- **Template** (committed): `.remote.example` — copy to `.remote` on every new box.
- **Loader module**: `scripts/setup/remote_config.py` (`from remote_config import load_config`).
  - Use `cfg.ssh()`, `cfg.ssh_T()`, `cfg.scp()`, `cfg.scp_download()`, `cfg.remote_path()`, `cfg.python()`, `cfg.python_carla()`, `cfg.run_ssh()`.
  - Never import `paramiko` in setup/scratch scripts; use `subprocess` + `cfg.ssh_T()` for stdin piping.
  - Do not add `HOST =` or `PORT =` module-level constants.

## 3. Sizing and Paths on the Box
- **Do not write a new `check_*.py` per run.** Use `python scripts/setup/box.py probe|status|env` and `python scripts/ops/run_status_report.py`.
- **Size jobs from `scripts/lib/machine.py`, never from `os.cpu_count()` / `multiprocessing.cpu_count()`.** Those count every host CPU, not the cgroup quota (`cpu.max`), and `pids.max` caps processes + threads (S-086). In shell: `eval "$(python3 scripts/lib/machine.py --export)"` gives `REC_DATALOADER_WORKERS`, `REC_EZV2_NUM_CPUS`, `REC_CARLA_LANES`, `REC_EVAL_OMP_THREADS`, `REC_BF16`. For EZ-V2 on Blackwell (`scripts/training/ezv2_setup_blackwell.sh`), always pass `EZV2_NUM_CPUS=<quota>` and `OMP_NUM_THREADS=1`.
- **Shell scripts source `scripts/lib/common.sh`** for paths (`WORKSPACE`, `MTHESIS_ROOT`, `GARAGE`, ...), `CUDA_DEVICE_ORDER=PCI_BUS_ID` (so CUDA indices match CARLA's `-graphicsadapter`), and the only copies of `kill_tree` / `clear_carla_ports`.
- **Several CARLA lanes on one GPU require `scripts/setup/patch_bench2drive.py`**, applied by `launch_b2d20.sh setup|run`.

---

## 4. SSH Key Safety, Heredoc Safety & Background Process Launching
- **Never Edit `authorized_keys` on Rented Boxes**: Editing `/root/.ssh/authorized_keys` once deleted the user's SSH key and locked out the box. Relay box-to-box copies through the local machine via `ssh A 'tar -cf - files' | ssh B 'tar -xf - -C dir'` and compare `sha256` on both ends.
- **Never Use Unquoted Bash Heredocs with Backticks or `$`**: Passing Python/Markdown containing backticks or `$(...)` through `<<EOF` executes command substitutions in bash. Always write multi-line scripts to a file first or use single-quoted `<<'EOF'` with environment variables.
- **Never Run Remote Jobs in Foreground Under an SSH Timeout**: Running a remote command without `setsid nohup` under a local timeout kills the remote process when the SSH client times out. Furthermore, `cd DIR && setsid nohup ... &` over SSH can hang waiting for stdout to close; always launch detached jobs with a subshell redirecting all standard streams:
  `(cd DIR; setsid nohup CMD > LOG 2>&1 < /dev/null &)`
  In `while read` loops calling `ssh`, always pass `ssh -n` or `<&3` so `ssh` does not consume the loop's stdin.
- **Fail-Fast Multi-Phase Scripts**: Any script chaining dependent steps (`cache -> train -> eval`) must check exit codes and stop immediately on failure (`set -e` or `|| exit 1`), never just `echo "status $?"`.

---

## 5. Pre-Launch Dependency Closure & Post-Launch Verification
- **Sync Full Import Closure**: Before launching on a box, sync every package the entry point imports (e.g. both `atari_qwen/` AND `src/`). Re-ship any file patched since the last sync (`git diff --name-only`), and re-apply local patches after any tarball refresh.
- **Preflight Import Check Per Venv**: A box has separate venvs (`/venv/main` for training/Atari and `venv_carla` for CARLA leaderboard evaluation). Before launching unattended jobs, test-import the actual entry-point modules (`import train_wor`, `import leaderboard.leaderboard_evaluator`, `import bench2drive_agent`) in the target venv. In `venv_carla`, `py-trees` must be pinned to `0.8.3`. For Town11–13 routes, verify `AdditionalMaps_0.9.15.tar.gz` is imported (`https://downloads.carlasim.com/Linux/AdditionalMaps_0.9.15.tar.gz` if `tiny.carla.org` returns 403).
- **Verified Launch (`launch_checked.sh` & Command-Line Audit)**:
  - Launch remote jobs with `scripts/ops/launch_checked.sh HOST PORT LOG "CMD"` (waits 50s and verifies the PID is alive and the log has no `Traceback`/`Error`/`No module named`).
  - After every launch, inspect `ps aux | grep <trainer>` and the log for a flag-unique line (e.g. `[B37 audit]`, `[A28]`, `-quality-level=`) to confirm new flags were not silently dropped by wrapper scripts.

---

## 6. Process Termination, CARLA Lane Queues & Classifier Denials
- **No Self-Kill `pkill -f`**: Never put a `pkill -f <pattern>` inside an SSH command or script whose own command line contains `<pattern>` (even `[h]` bracket tricks fail if the filename appears elsewhere in the same command). Kill by PID, session ID (`pkill -s <sid>`), or guardian process tree (`ps --forest`).
- **Replacing Launchers / Queues**: Killing a queue/launcher script (`launch_boxA.sh`, `queue_A9c.sh`) leaves its already-spawned training/eval children running! Before starting a replacement queue, check `/proc/PID/cmdline` for running `train_wor` / `b2d_guardian` / `CarlaUE4` processes, kill orphaned `CarlaUE4` servers holding RPC ports (`-carla-rpc-port=`), and never run two chains with the same run label.
- **CARLA Eval Lane Health & Route Files**:
  - A CARLA eval lane is healthy **only** if its `.json` record count/mtime advances and `CarlaUE4` holds VRAM (~4–5.3 GB/lane); live guardian processes alone can be hung on orphaned evaluators.
  - Keep route list files strictly LF-only (`tr -d '\r'`); Windows CRLF (`\r`) causes `"Couldn't find the route with id"` and false crash exclusions.
  - Before relaunching a failed lane, move old `guardian_logs/<label>*` and poisoned `bench2drive_out/<label>*` aside or use a new label (`<label>d<k>` via `scripts/eval/lane_plan.py`) so stale attempt counts do not trigger false crash-route skips.
- **Windows Git-Bash Background Loops**: Stopping a background task does **not** stop child `bash.exe` sync loops (`backup_run_pairs.sh`, `pair_loop`), which will race with replacement loops on `.part` files. Always check `Get-CimInstance Win32_Process | Where-Object CommandLine -match 'backup_run_pairs|pair_loop|sync'` and stop old PIDs via `Stop-Process` before starting a new loop.
- **Rejected Launches & Permission Denials**:
  - If the user rejects or cancels a launch tool call, check the target boxes first (`ps`, lane file mtimes) before answering — background steps may have already started. Report what is running and ask whether to stop it.
  - If a safety/permission classifier denies a remote launch (`Modify Shared Resources`) or config edit (`Self-Modification`), **never** retry or route around it. Report the exact command/rule to the user and wait for their explicit go or let the user run it directly.
