---
trigger: always_on
description: Secret handling (never read dotfiles), daily API/Google Drive token rotation, Vast.ai Secretary Problem rental rule, provisioning probes, token push rules, and 10-minute Athens-time status reports.
---

# Secret Handling, Daily Key Rotation & Vast.ai Fleet Operations

## 1. Strictly Never Read `.env` or Dotfiles (`.*`)
- **NEVER** use file-reading tools, `grep`, or shell commands (`cat`, `type`, `Get-Content`, `head`, `tail`, etc.) to directly read any `.env` files or any files starting with a dot `.` (e.g. `.env`, `.env.local`, `.vast_api_key`, `.remote`, `.gdrive_rclone.conf`, `.credentials.json`).
- **Always assume credentials in dotfiles exist and work correctly.** If any information or verification is needed regarding them, **ask the user directly**.
- **Google Drive Token (`.gdrive_rclone.conf`)**:
  - Never read, cat, grep, or print `rclone config show` / `rclone config dump` output. Test only via commands that print no token (`rclone lsd`, `rclone check`, or the sync scripts).
  - Rotation: `bash scripts/ops/rotate_gdrive_token.sh` (mints new token, ships to running boxes, revokes old one; optional daily task: `scripts/ops/schedule_gdrive_rotation.ps1`). At the start of each session, remind/check with the user whether the daily rotation ran.
  - Never leave a valid token on an abandoned box that stays alive; revoke via rotation when giving up a live box.

## 2. Daily API Key Rotation Reminder
- At the start of every new calendar day / daily session, **prompt the user to reroll/rotate their API keys** (Vast.ai, Hugging Face, Google Drive token) to maintain zero-leak security hygiene.

## 3. Token Push Commands (User Runs Them — Never the Agent, 2026-10-28)
- Once a new box is connected and passes the E: and HF provisioning checks, **output the exact token push commands for the user to run, and NEVER run them yourself**:
  1. `scripts/ops/push_gdrive_token_to_boxes.sh PORT HOST` (only after the E: and HF checks pass)
  2. `scripts/ops/push_hf_token_to_boxes.sh PORT HOST`
  (using the box's SSH `PORT` and bare `HOST`; `box_provision_probe.sh` prints both lines at its end).

---

## 4. Vast.ai Fleet Access & The Secretary Problem Rental Rule (2026-10-28)
- **CLI Commands**:
  - Inspect fleet: `python scripts/setup/vast_fleet.py list` or `vastai show instances` (check `vastai show user` for credit balance).
  - Target an instance in `.remote`: `python scripts/setup/vast_fleet.py select <idx_or_id>`.
  - Auth is pre-configured in `~/.config/vastai/`; no tokens are needed in CLI arguments.
- **Hard Offer Filters (Every Candidate Must Pass All)**:
  1. **Bandwidth Cost (< $8/TB both ways, 2026-10-10)**: `inet_up_cost * 1024 < 8` and `inet_down_cost * 1024 < 8` (offer fields are $/GB, so `<= 0.0078`). **Print both upload and download $/TB costs for every candidate before renting.**
  2. **Host Verification**: Never rent `"deverified"` hosts (`verification != deverified`; unverified is fine). Reliability `>= 0.97`, disk `>= 200 GB`.
  3. **CPU Model**: Prefer AMD Ryzen (5000/7000/9000) or AMD EPYC over older Intel Xeon E5 (Atari runs ~35% slower and CARLA is CPU-bound on Xeon E5).
  4. **GPU & VRAM**: At least 24 GB VRAM per GPU and Ampere or newer for native `bf16`; prefer 32 GB cards (e.g. RTX 5090) for `30m` runs (which peak at ~18.1 GB at batch 256).
  5. **2x and 4x GPU Boxes (2026-10-28)**: Include in the pool **only** when the queue has at least one independent run per GPU, each pinned with `CUDA_VISIBLE_DEVICES=k` (one trainer per GPU in `nvidia-smi`), with `>= 8` CPU cores and `>= 32 GB` RAM per GPU, and `$/GPU-hour` no higher than the single-GPU benchmark.
- **Secretary Problem Selection Protocol (37% Rule, 2026-10-28)**:
  1. Collect the filtered pool of $n$ offers (search again if $n < 5$).
  2. **Shuffle** the pool with a recorded random seed (since price-sorted order violates random arrival).
  3. **Exploration (First $\lceil 0.37 n \rceil$ offers)**: Rent nothing. Record the sample's price/score range and the **benchmark**: the highest score in the sample, where:
     `Score = (GPUs * bf16_dense_TFLOPS_per_GPU) / dph_total`
  4. **Exploitation (Remaining offers in shuffled order)**: Rent the **first** offer whose score strictly beats the benchmark, and stop searching immediately. If none beats it, rent the benchmark offer if still available, else re-search.
  5. Record pool size, shuffle seed, sample size, benchmark offer/score, and chosen offer in the rental note.
  6. **Duds**: `vastai create instance` can return `"success": false` and still create an instance — always check `vastai show instances` after renting. If an instance never boots within ~12 min, refuses SSH keys, or fails the provisioning probe, destroy it immediately; a dud does **not** reset the benchmark (continue with the next offer in the shuffled list).
- **Destroy Rule (2026-10-17)**: Destroy a box **without asking** when no progress would be lost (no running job, no unsaved results). Always ask and complete verified E: + Drive/HF saves before destroying any box with unsaved work. Report every destroy with instance ID and reason.

---

## 5. Mandatory Provisioning Probe & 10-Minute Status Reports (2026-10-28)
- **Provisioning Probe (`box_provision_probe.sh`)**:
  - Run `bash scripts/ops/box_provision_probe.sh PORT root@HOST` as the very first command once SSH answers, before shipping any job or token.
  - Checks: (a) GPU model, VRAM, and foreign GPU load (any pre-existing GPU process = destroy; note RTX 5090 driver 595.84 idle reading artifact can be checked with a 20s bf16 matmul benchmark ~235 TFLOPS); (b) cgroup `nproc` and `pids.max`, CPU model, RAM, disk; (c) Hugging Face DNS (must not resolve to Facebook/censored IP), HEAD request, and 200 MB download (`>= 10 MB/s`); (d) box -> `E:` 200 MB SSH pull (`>= 2 MB/s`); (e) after the user pushes the Drive token, box -> Google Drive 200 MB upload (`>= 2 MB/s`). For CARLA boxes, also check `ls /dev/nvidia* | wc -l` (`< 14` to avoid Vulkan hang) and `XDG_RUNTIME_DIR=/tmp vulkaninfo --summary`.
- **10-Minute Run Status Report (`run_status_report.py`)**:
  - Every 10 minutes while jobs run, execute `python scripts/ops/run_status_report.py` (`--ssh root@HOST --port P` for a box; repeat `--label/--log/--run` per run; `--gpus --credit` on a box).
  - Report per run in **Athens time (`Europe/Athens`)**: (1) error count and last error line; (2) env steps, updates, `return(20)`, and ETA in Athens time with hours left (computed from remaining steps / measured rate); (3) newest checkpoint update, age, whether `checkpoint_latest.pt` + `replay_latest.npz` are a matched pair, and replay size; (4) last `E:` backup line; (5) last two log lines; plus per-box GPU load/VRAM/processes and Vast credit.
  - Flag immediately: any error/Traceback, no log write for 20 min, checkpoint older than 30 min, unmatched pair, or an idle paid GPU.
