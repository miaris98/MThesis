# Box operations (S-115, 2026-10-03)

Tools used on rented Vast.ai boxes; every one was run for real that day. HF token only ever lives in `/dev/shm/hf_token` on the box
(sent over stdin, never printed). Never `pkill -f` a pattern in the same ssh command that relaunches a matching script.

| Script | Where | What |
|---|---|---|
| `sync_forever3.sh PORT HOST LOCALDIR "PATHS" [SEC]` | home (bash) | mtime-aware box -> E: backup loop; run as a background task and **restart before 2 h** (the task dies silently); check `.sync_status.log` times |
| `push_verify.py LOCAL_ROOT HF_SUBDIR` | box | upload a folder to the HF relay, then sha256-check every file against HF's LFS records (nested subdirs OK) |
| `finisher.sh HF_SUBDIR "[p]attern" PATH...` | box | unattended: push results hourly while jobs matching the pattern run, once more when they end, "FINISHED" in `/workspace/finisher.log` |
| `provision_eval.sh` | box | eval-only CARLA box: CARLA + venv_carla + checkpoints from HF (sha-checked) + `PROVISION_DONE` marker |
| `launch_lanes_nokill.sh BOX LANEFILE` | box (stdin) | start one `queue_A9c.sh` lane per `BOX PORT GPU JOBS...` line; stops stray evaluators first |
| `followon.sh BOX CAP NGPU PORT0 ROUTE...` | box | single-route lanes started as GPU slots free up (CAP lanes per GPU) |
| `gen_rep.py` | home | builds the S-115 repeat-eval lane files (60 routes x E15 a/b + arm J seeds 0-3) |
| `restore_b35_checkpoints.py` | box | the 16 B35 env30000 checkpoints from HF (`b35_*`, `b35_100k_env30000/`), sha-checked, + a 200 MB HF upload speed test |
| `run_b24_P.sh` | box | B24: 30 episodes at 64 sims on the 16 checkpoints, one evaluator each over 4 GPUs (add `--graph`) |
| `profile_search.py`, `tree_depth.py`, `test_graphed.py` | box, repo root | B23: search profile, tree-depth distribution, graphed-vs-eager equivalence + speed |

First command on any new box (S-115, box N): `getent ahosts huggingface.co; curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co`.
