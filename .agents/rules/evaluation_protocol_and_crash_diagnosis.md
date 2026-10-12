# Evaluation Protocol & Crash Diagnosis Rules

Written after S-137 / S-138 (2026-10-09): our CARLA evals ran at `-quality-level=Low` since setup (training data and the official Bench2Drive evaluator use Epic), that setting made all 11 Town03 routes segfault at load, and an isolated retest of the crashing configuration was generalised into "these routes are broken".

## 1. The evaluation protocol is part of the result
- **CARLA rendering must match the training data and the benchmark.** carla_garage data collection and Bench2Drive's `leaderboard_evaluator.py` launch CARLA with no quality flag (Epic). Any number compared with TF++, a published Bench2Drive score, or used as a thesis claim is measured at Epic (`CARLA_QUALITY=Epic` as a CKE flag, read by `scripts/eval/run_bench2drive.sh`).
- Every server flag (`-quality-level`, `-vulkan`/`-opengl`, fps, resolution, sensor settings) is protocol. Record it in the run label or run config; never pair runs that differ in it.
- Before adding or keeping any non-default simulator flag, check what the training data and the official evaluator use, and measure its effect with a paired run. Write the reason into the script and the challenges log.
- **Atari:** a number set against a paper uses the paper's protocol. EZ-V2's 400.1 (Atari-100k): no sticky actions, up to 30 no-op starts, 27,000-step cap. Sticky actions (p = 0.25) are a separate robustness column and never compared with the paper's number. State protocol (sticky, no-ops, simulations, depth cap, episodes, distinct games) next to every score.

## 2. A route is not "broken" until shown to be
Before excluding a route from any run (a9_exclude, `gen_lanes.py` CRASH set, lane lists):
1. **Where does it crash?** Read the last non-tick lines of the attempt log. Before the first game tick (`> Loading the world`) = map / scenario load; after = driving-dependent.
2. **Has it ever completed?** Search the E: archive (`live_*/**/*.json` records) for that route under any agent or flag. One past completion means the route is not intrinsically broken.
3. **Vary one factor at a time on the same host:** plain checkpoint vs switches, another agent (original WoR), another host, Low vs Epic. A retest only tells you about the configuration it ran.
4. **Load-time crash:** run `scripts/eval/run_town_probe.sh <Town>` with the eval's exact server flags, then with the flag in question changed.
5. Exclusions are per (agent, switches, rendering, host). An `a9_exclude.txt` entry never goes into another arm's route list or the global CRASH set without steps 1-4.
- `Signal 11`, `Malloc Size=65538 LargeMemoryPoolOffset=65554` and `CommonUnixCrashHandler` are UE4's generic crash handler, not a cause.
- Report coverage with every DS: routes driven / attempted, and which routes were crash-skipped for that arm.

## 3. Operations
- A paid box with 0 running lanes is idle: the 10-minute check must flag it and the next prepared job starts within one tick.
- Periodic status reports: use a Monitor loop (each line is an event) and re-arm it at every expiry; session cron jobs only fire while the session is idle and did not deliver on 2026-10-09.
- Sync scripts: check the pulled file count against what the box has after the first run (the 2026-10-09 Atari sync pulled logs only because of a wrong path).
- Hugging Face uploads run on the box with its `/dev/shm/hf_token` (relay it box-to-box); never read `.env` or other dotfiles locally (see `secret_handling_and_daily_rotation.md`).
