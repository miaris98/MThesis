---
trigger: always_on
description: Operational rules for Athens timezone reporting, git commit attribution (no Claude/AI trailers), gitignored local folders, script triage, claim verification, and budget planning.
---

# Workflow, Git Attribution, Timezone & Verification Rules

## 1. Report All Times and ETAs in Athens Time (`Europe/Athens`)
- The user works in **Athens time (`Europe/Athens`: EET = UTC+2 in winter, EEST = UTC+3 in summer)**. Remote Vast.ai boxes log in UTC.
- Convert every timestamp, schedule, and ETA shown to the user into **Athens time** and label it explicitly (keep raw UTC only when quoting log lines verbatim).
- Note: In Windows Git Bash, `TZ=Europe/Athens date` lacks the tz database and wrongly prints UTC; use plain `date` (which uses Windows local Athens time) or PowerShell `Get-Date`.

---

## 2. Git Commits & Pull Requests: User Identity Only, No AI Attribution (2026-10-12)
- All commits must be authored under the user's configured git identity (`dionushs`).
- **NEVER add `Co-Authored-By: Claude ...` (or any AI co-author trailer) to commit messages, and NEVER add `"Generated with Claude Code"` (or similar) to PR bodies.**
- After committing, verify with `git log -1 --format=%B`.
- **Push only when the user explicitly asks.** Never rewrite published git history or force-push without an explicit instruction from the user (since docs and logs cite commit hashes like `049da8e`).

---

## 3. Gitignored Local Folders & Repository Hygiene (2026-10-11)
- The root `.gitignore` ignores `/docs/`, `/.antigravity/`, `.agents/agents/`, and `Thesis_typist/` as local-only working directories.
  - **Do not** `git add -f` or commit new files under `/docs/`, `/.antigravity/`, `.agents/agents/`, or `Thesis_typist/`.
  - The 37 legacy files already tracked under `docs/` (such as `docs/todo/TODO_ACTIVE.md`) remain tracked unless the user explicitly asks to untrack them.
  - **Never** add a blanket `.gitignore` rule for all of `.agents/` — `.agents/rules/`, `.agents/skills/`, and `.agents/workflows/` are shared and tracked.
- **Scripts & Scratchpad Triage (2026-10-11)**:
  - Keep run logs, reusable tools, and run manifests (e.g. `fleet_monitor/lanes/`).
  - Before deleting any one-off script or scratch file, document one-off scripts verbatim in `docs/ops/`, verify that any scratch artifact has a byte-identical `sha256` match on `E:\MThesis_EXP`, and check that no running background process is reading the file (e.g. `boxes.txt`).
  - Ask the user before deleting any git-tracked script.

---

## 4. Verify Every Summary Claim and ETA Against Raw Logs (2026-10-11 / 2026-10-19)
- **Never Trust Unverified Summaries or Handoffs**: Before putting any metric, speed-up, or fleet status into the thesis, `TODO_ACTIVE.md`, or a decision, verify it against the raw log, JSON record, or script output and recompute the number. Forward-only microbenchmarks are not training throughput, and speed-ups count only when outputs match on the same GPU model.
- **Thesis Audit & Typst Rules (`Thesis_typist/`)**:
  - Every empirical claim or figure in `Thesis_typist/` must trace to a verified log entry (`S-NNN` in `challenges/`) or `docs/design/what_has_worked_2026-10-04.md`.
  - When running Python tools on `Thesis_typist/` (e.g. `fetch_references.py`, `extract_citations.py`, `build_fact_check.py`), use `py -I` for scripts reading downloaded PDFs, and write patch scripts as files rather than bash heredocs.
- **Honest ETA Calculation**:
  - Before quoting or defending a CARLA or Atari evaluation ETA, diff the exact workload config (route XML file, town, route count, scenario density, GPU model, concurrency) against historical runs (`meta.duration_system` / `meta.duration_game` in leaderboard JSONs). Remember that a better driving model drives further before terminating, taking more wall-clock time per route, and `routes_validation.xml` routes 10–19 are adverse-weather repeats of routes 0–9.
  - For CARLA lane queues, the job ETA is governed by the **slowest lane** (`scripts/ops/lane_eta.py`), not aggregate routes/hour (~104 runs/h per 12-lane box; ~5 CPU cores and ~5.3 GB VRAM per Epic lane).
  - For Atari sticky/paper evaluations on a shared GPU, total vector steps/s is fixed per GPU (~4.7 vector steps/s on RTX 3090; `rate * concurrent_jobs ≈ const`), and a single stuck 27,000-step game determines wall time (~1.6 GPU-h). Always compute `ETA = remaining_steps / measured_per_job_rate` and note that per-job rate rises as sibling jobs finish.

---

## 5. Budget Planning, Queue Ordering & Idle GPU Fillers
- **Budget & Measured Rates**: Quote costs in both **USD (`$`) and EUR (`€`)** (stating the conversion rate) and calendar estimates as **P50 / P90** in session-days (~5.5 productive box-hours/day, plus ~1.0–1.4h provisioning, 12% CARLA reruns, and the >=30-min end-of-day save margin).
- **Prioritise New Tests Over Re-Verification (2026-10-12)**: When ordering a box queue, schedule new experiments and speed tests ahead of parity re-checks or claim reproductions (which can run later or as fillers), while still running the fast 1-update numerical exactness check for any new speed optimization.
- **Start Prepared Filler Queues When GPUs Go Idle (2026-10-05)**: Keep a prepared eval-only filler queue ready (`scripts/ops/gen_lanes.py` repeats, switch-only evals) and launch it immediately whenever a rented GPU finishes its main job and goes idle, without waiting to ask, provided it never delays the end-of-day save. (If the environment's permission classifier blocks remote launches, report the idle GPU and exact filler command immediately for the user's go).

