# Box operations (S-115, 2026-10-03; S-126 / S-127, 2026-10-05)

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
| `gen_lanes.py ROUTE_DUR.json OUT BOX:GPUS:VRAM_GB:CPUS [--routes lead_obstacle\|all\|FILE] ARM...` | home | S-126: lane files sized from the box (**lanes per GPU = floor(VRAM / 5.5 GB)**, <= CPUS / 5 in total, <= 12, each job >= 8 routes), arms `LABEL=CKPT` (CKL) or `LABEL=CKPT\|K=V,...` (CKE, env switches such as `WOR_CONTACT_REFLEX=1`; a `;` inside a value stands for `,`), balanced by the A9 per-route wall times |
| `gen_lanes_a36.py ROUTE_DUR.json OUT BOX:GPUS:LANES_PER_GPU LABEL=CKPT...` | home | S-126: the first lane generator (A36 merges on the 103 lead + obstacle routes); `gen_lanes.py` supersedes it |
| `provision_train_box.sh` | box | S-126: training box for the pooled cache: arm J's data (6 towns + the Town12/13 obstacle archives; **>= 400 GB disk**, the download is the long pole: ~5 h at 11-12 MB/s), TF++ weights, E e15 from the relay, then `build_pooled_cache.py` per GPU shard; markers in `/workspace/train_prov/` |
| `run_head_arms.sh GPU LABEL SEED [train_wor.py flags]` | box | S-126: one head-only arm on the pooled cache (J's recipe from E e15, epochs 16-20, ~12-15 min); `jc_s0` is the control that must match J; checkpoints under `/workspace/checkpoints/<LABEL>/` feed `CKL` / `CKE` lane jobs |
| `run_b43_depth.sh "3 5 8 0" NPAR [SIMS]` | box | S-126: B43 step 1, the 16 restored B35 checkpoints at 64 simulations with the imagined depth capped (0 = unlimited reference) |
| `../atari/run_b37_probe.sh GPU NAME [flags]`, `../atari/run_b36.sh` (`EXTRA="..."`) | box | S-126: the B37 2,000-update probe (one setting per GPU) and the 30k screens with extra trainer flags; **after every launch read the process command line and a log line unique to the new flag** (S-126: SGD ran for 20 min) |

First command on any new box (S-115, box N): `getent ahosts huggingface.co; curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co`.

**Saving a box over a slow link (S-119).** Ask the destroy time first and start the save >= 90 min before it, sized with a measured uplink (box -> home over ssh and a 200 MB upload to HF; box 3 of 2026-10-04 did 2 MB/s and 0.2 MB/s).
(1) gzip the replay buffers on the box (`ssh box "gzip -1 -c replay_latest.npz" > replay_latest.npz.gz`): uint8 frames shrank 28x (817 MB -> 29 MB). (2) Record sha256 on the box, copy, then compare; a trainer that saves in between makes an early hash stale, so re-hash
on the box after copying (compare `gzip -dc file.gz | sha256sum` for the gzipped ones). (3) Never start a broad overwriting `tar` in the last minutes: an interrupted one truncated a checkpoint (it only had to copy the files whose size or mtime differs). (4) Do not report a box as safe
until the sha pass over *all* checkpoints, resume files and small files has finished; list what is missing instead. `provision_eval.sh` needs `/workspace/pcla_wor.tgz` for the original WoR (`WORL:` jobs).

**What 2026-10-05 added (S-126, S-127).** (1) *Before renting a CARLA box:* `vulkaninfo` must work (a host with ~14 `/dev/nvidia*` nodes hung at graphics init) and lanes <= VRAM / 5.5 GB (8 lanes on 24 GB = OOM). (2) *Unreachable is not destroyed:* box 4's proxy refused connections twice for ~25 min and came back intact; retry proxy and direct for 10-25 min before acting; `sync_forever3.sh` writes `ERROR empty remote listing (box may be gone)` and never claims a pass. (3) *Sync lists name every output folder* (`guardian_logs` was missing; found only by the final hash comparison). (4) *The save:* sha256 of every box file (`find ... -print0 | xargs -0 -P4 sha256sum`, run on the box) against the E: copy by path, then the HF listing sizes (`HfApi.list_repo_tree`) against E:; list what is left behind on purpose (replay buffers, `checkpoint_latest.pt`). (5) `arm_report.py` takes explicit roots: a glob that includes an archived `_INVALID_*` folder adds its crashed records as "infrastructure failures". (6) Keep a prepared eval-only filler queue so no GPU idles when the main jobs end (the afternoon of 2026-10-05 wasted ~$4-5).
