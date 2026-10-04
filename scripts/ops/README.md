# Box operations (S-115, 2026-10-03)

Tools used on rented Vast.ai boxes; every one was run for real that day. HF token only ever lives in `/dev/shm/hf_token` on the box
(sent over stdin, never printed). Never `pkill -f` a pattern in the same ssh command that relaunches a matching script.

| Script | Where | What |
|---|---|---|
| `sync_forever3.sh PORT HOST LOCALDIR "PATHS" [SEC]` | home (bash) | mtime-aware box -> E: backup loop; run as a background task and **restart before 2 h** (the task dies silently); check `.sync_status.log` times |
| `push_verify.py LOCAL_ROOT HF_SUBDIR` | box | upload a folder to the HF relay, then sha256-check every file against HF's LFS records (nested subdirs OK) |
| `finisher.sh HF_SUBDIR "[p]attern" PATH...` | box | unattended: push results hourly while jobs matching the pattern run, once more when they end, "FINISHED" in `/workspace/finisher.log` |
| `provision_atari_box.sh [restore]` | box | Atari box: **HF reachability check first**, deps, the B15 diagnostics state file (HF `diag/b15_states_512.npy`), optionally the 16 B35 checkpoints |
| `provision_eval.sh` | box | eval-only CARLA box: CARLA + venv_carla + checkpoints from HF (sha-checked) + `PROVISION_DONE` marker |
| `launch_lanes_nokill.sh BOX LANEFILE` | box (stdin) | start one `queue_A9c.sh` lane per `BOX PORT GPU JOBS...` line; stops stray evaluators first |
| `followon.sh BOX CAP NGPU PORT0 ROUTE...` | box | single-route lanes started as GPU slots free up (CAP lanes per GPU) |
| `gen_rep.py` | home | builds the S-115 repeat-eval lane files (60 routes x E15 a/b + arm J seeds 0-3) |
| `restore_b35_checkpoints.py` | box | the 16 B35 env30000 checkpoints from HF (`b35_*`, `b35_100k_env30000/`), sha-checked, + a 200 MB HF upload speed test |
| `run_b24_P.sh` | box | B24: 30 episodes at 64 sims on the 16 checkpoints, one evaluator each over 4 GPUs (add `--graph`) |
| `profile_search.py`, `tree_depth.py`, `test_graphed.py` | box, repo root | B23: search profile, tree-depth distribution, graphed-vs-eager equivalence + speed |
| `gen_lanes_0410.py` | home | S-119: CARLA lane files for the original WoR x2 on the 60 repeat routes + arm J on the other routes, balanced by the A9 per-route wall times (`BOX:GPUS:LANES_PER_GPU`) |
| `run_eval_queue.sh NPAR` | box | S-119: a bounded queue of deferred Atari evaluations from `/workspace/eval_jobs.txt` (B24 rest, B2 sticky), round-robin over the GPUs; keep NPAR <= 3 next to training (8 slowed the trainers 1.7x) |

First command on any new box (S-115, box N): `getent ahosts huggingface.co; curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co`.

**Saving a box over a slow link (S-119).** Ask the destroy time first and start the save >= 90 min before it, sized with a measured uplink (box -> home over ssh and a 200 MB upload to HF; box 3 of 2026-10-04 did 2 MB/s and 0.2 MB/s).
(1) gzip the replay buffers on the box (`ssh box "gzip -1 -c replay_latest.npz" > replay_latest.npz.gz`): uint8 frames shrank 28x (817 MB -> 29 MB). (2) Record sha256 on the box, copy, then compare; a trainer that saves in between makes an early hash stale, so re-hash
on the box after copying (compare `gzip -dc file.gz | sha256sum` for the gzipped ones). (3) Never start a broad overwriting `tar` in the last minutes: an interrupted one truncated a checkpoint (it only had to copy the files whose size or mtime differs). (4) Do not report a box as safe
until the sha pass over *all* checkpoints, resume files and small files has finished; list what is missing instead. `provision_eval.sh` needs `/workspace/pcla_wor.tgz` for the original WoR (`WORL:` jobs).
