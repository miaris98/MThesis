<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**This project has a knowledge graph. Start with the code-review-graph
MCP tools to narrow scope, then read the source.** The graph is cheaper than scanning files and
gives you structural context (callers, dependents, test coverage) that file search cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

### Verify in the source

- Narrow scope with the graph, then read the source. Do not change code from graph output alone.
- For any non-trivial change, read the implementation and the relevant tests before concluding.
- Verify the exact source when touching behavior, database logic, migrations, retries, fallbacks,
  recovery, or compatibility code.
- When the graph and the source disagree, the source wins. The graph may be stale or may not
  model that relationship.
- An empty graph result can mean "not indexed" or "not statically visible", not "does not exist".

### Key Tools

| Tool | Use when |
| ------ | ---------- |
| `detect_changes_tool` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context_tool` | Need source snippets for review — token-efficient |
| `get_impact_radius_tool` | Understanding blast radius of a change |
| `get_affected_flows_tool` | Finding which execution paths are impacted |
| `query_graph_tool` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes_tool` | Finding functions/classes by name or keyword |
| `get_architecture_overview_tool` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes_tool` for code review.
3. Use `get_affected_flows_tool` to understand impact.
4. Use `query_graph_tool` pattern="tests_for" to check coverage.
<!-- /code-review-graph MCP tools -->

## Knowledge graph routing: code-review-graph vs graphify

Two graphs are maintained for this repo — pick by question shape, don't ask
both for the same thing:

- **code-review-graph (CRG)**: symbol-precise. Use for "where is X defined",
  "what calls Y", blast-radius/impact of a change, test coverage for a
  function. SQLite-backed, sub-second incremental updates, auto-refreshes on
  every Edit/Write via the PostToolUse hook.
- **graphify** (`graphify-out/`): concept-precise. Use for cross-document
  questions that span code + the `.md` docs/TODOs/rules (e.g. "why was WoR
  chosen as the baseline", "what's the rationale behind X design decision",
  "how do the CARLA setup guides relate to the training pipeline"). Query
  with `graphify query "<question>"`. Rebuild manually with `/graphify --update`
  after doc/rule changes — it does not auto-update like CRG.

A `PreToolUse` hook (`~/.claude/scripts/smart-grep-hook.sh`) intercepts
`Grep` tool calls and raw `grep`/`rg`/`find` in Bash: if either graph already
has an answer for the search term, it shows that answer instead of letting
the search run blind, and blocks a second identical search in the same
session (append `--graph-tried` to a Bash command to force it through).

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

### External Experiments Knowledge Graph (E:\MThesis_EXP)

The external drive `E:\MThesis_EXP` contains historical checkpoints, Bench2Drive evaluations, MLflow runs, and experiment logs, indexed with graphify at `E:\MThesis_EXP\graphify-out\`.

Rules:
- For questions about past experiments, historical checkpoints, Bench2Drive scores, or external drive logs, first consult `E:\MThesis_EXP\graphify-out\wiki\index.md`.
- To query the external graph: `graphify query "<question>" --graph "E:\MThesis_EXP\graphify-out\graph.json"`.
- Maintenance: When new runs, checkpoints, or evaluations are synced to `E:\MThesis_EXP`, maintain the external graph by running:
  `python scripts/sync/build_external_graph.py`
  or re-exporting the wiki:
  `graphify export wiki --graph "E:\MThesis_EXP\graphify-out\graph.json" --labels "E:\MThesis_EXP\graphify-out\.graphify_labels.json"`

## Experiment Tracking, Two-Stage Syncing & Compute Utilization

### 1. Mandatory MLflow Experiment Tracking
Whenever creating or updating scripts for training, testing, or evaluating:
- **Always integrate MLflow experiment tracking**.
- Log all hyperparameters/config, step/epoch metrics (losses, ADE, lateral error, rewards, episode return, SPS), and final artifacts/checkpoints.
- Preserve local and external tracking store compatibility (`paths.mlruns_dir()` with `MLFLOW_ALLOW_FILE_STORE=true`).

### 2. Two-Stage Syncing Pipeline (External First -> Hugging Face Second)
All artifacts, checkpoints, and evaluation results must follow a strict two-stage sync pipeline:
1. **Stage 1 — External Archive First (`E:\MThesis_EXP`)**:
   - Sync all logs, telemetry CSVs, model weights, and MLflow runs to `E:\MThesis_EXP` first using `scripts/sync/sync_experiments.py` or direct export.
   - Update the external knowledge graph and catalog via `python scripts/sync/build_external_graph.py`.
2. **Stage 2 — Google Drive Second (replaces Hugging Face, user 2026-10-19)**:
   - Push verified champion checkpoints, evaluation summaries, resumable pairs and gameplay/rollout videos to Google Drive (`MThesis_backups/`) with rclone, from the boxes, via `scripts/sync/gdrive_push_dir.sh DIR SUB` (checksum-verified, prints `PUSH_VERIFY`). Full guide: `docs/ops/google_drive_sync.md`.
   - Credential: `.gdrive_rclone.conf` (repo root, gitignored; holds the OAuth refresh token). Ship it to the boxes with `scripts/ops/push_gdrive_token_to_boxes.sh` (lands in `/dev/shm/gdrive_rclone.conf`, RAM only). Never put it in a command line, a log or the chat.
   - Migration status 2026-10-19 (evening): `fleet_monitor/sync_boxes.sh` and `finisher.sh` push to Drive by default (`STAGE2=hf` / `FIN_TARGET=hf` for the old relay); `live_sync_loop.sh` is stage 1 only. Not yet run on a Vast.ai box. Do not claim a backup is on Drive until a box-side `PUSH_VERIFY ... bad 0 errors 0` line has been seen.
   - Hugging Face scripts stay for restoring old runs; do not delete them.

### 3. Compute Utilization & Parallelism (GPU & CPU Co-Utilization)
Before triggering any training or evaluation run:
- **Mandatory GPU & CPU Co-Utilization**:
  - **GPU**: Fully saturate available GPU compute and VRAM on the target instance (e.g., RTX A4000 16GB, RTX 3090 24GB, 2x Tesla T4), sizing batch sizes and model contexts to maintain a safe ~10–15% VRAM headroom to prevent CUDA OOM crashes.
  - **CPU**: Never let system CPU cores sit idle during training or simulation. Fully leverage high-core CPUs (e.g., AMD EPYC 64-core on Vast.ai).
- **Default to Parallel Execution**:
  - **Vectorized Environments**: For RL / simulation workloads (Atari, CARLA), always vectorize environment stepping (`num_envs=16`, `32`, or `64` scaled to available CPU cores).
  - **Parallel Data Loaders**: In supervised or offline distillation (e.g., PDM-Lite / World-on-Rails), scale PyTorch DataLoader workers (`num_workers=4` to `8`+), enable `pin_memory=True`, and set `prefetch_factor=2` to ensure GPU compute is never starved by CPU data loading.
  - **Parallel Evaluations & Sweeps**: Run evaluation rollouts, seed replications, and hyperparameter trials in parallel across available ports/threads.
  - **No Sequential / Single-Threaded Execution**: Never default to single-threaded or sequential execution (`num_envs=1`, `num_workers=0`, or sequential seed loops) unless the user explicitly instructs to run sequentially or single-threaded for isolated debugging.

## Secret Handling & Daily API Key Rotation

- **NEVER directly read any `.env` files or files starting with `.` (dotfiles)**:
  - Do not use file viewing tools, grep, or shell commands to read `.env`, `.env.local`, `.vast_api_key`, `.remote`, etc.
  - Assume all credentials and dotfiles work correctly.
  - If you need any information regarding them, ask the user directly.
- **Daily Key Rotation Reminder**:
  - Prompt the user to reroll/rotate API keys (Vast.ai, Hugging Face, etc.) every day / at the start of each daily session.
- **Google Drive token (`.gdrive_rclone.conf`)**:
  - It is a dotfile: never read it, cat it, grep it or print `rclone config show` / `rclone config dump` output. Test it only through commands that print no token (`rclone lsd`, `rclone check`, the scripts).
  - Rotation = `bash scripts/ops/rotate_gdrive_token.sh` (new token, ship to every running box, then revoke the old one; needs one browser click). Optional daily task: `scripts/ops/schedule_gdrive_rotation.ps1`. Check `.gdrive_rotation.log` at the start of each session and tell the user if the last rotation failed or did not run.
  - After any new box is rented, or one is lost or reused, run `bash scripts/ops/push_gdrive_token_to_boxes.sh` (or rotate). Never leave the token on a box that is being given up: it is in `/dev/shm` and dies with the box, but a box that is kept alive and abandoned should be revoked via a rotation.

## Vast.ai Platform Integration & Autonomous Access

Claude can now query and manage the Vast.ai fleet directly via CLI tools:
- **CLI Command**: `vastai` (available via `python scripts/setup/vast_fleet.py list` or `vastai show instances`).
- **Authentication**: Pre-configured in user profile (`~/.config/vastai/` with active 2FA session). No tokens needed in commands.
- **Inspect Fleet**: `python scripts/setup/vast_fleet.py list` probes all active instances (direct and proxy SSH).
- **Target an Instance**: `python scripts/setup/vast_fleet.py select <idx_or_id>` automatically writes the selected box IP and port into `.remote`.
- **Search Offers**: `vastai search offers 'gpu_name=RTX_3090 num_gpus>=1 ...'`.
- **Rental rule (user, 2026-10-10)**: only rent offers whose internet traffic costs are below **$8 per terabyte in both directions** (`inet_up_cost * 1024 < 8` and `inet_down_cost * 1024 < 8`, the offer fields are $/GB). Never rent "deverified" hosts (unverified is fine). Print both costs for every candidate before renting and check the box's CPU model (Ryzen/EPYC over Xeon E5).
- **Secretary rule for renting (user, 2026-10-28)**: from the filtered offer pool (the rules above, plus at least 24 GB VRAM per GPU, 32 GB for 30m runs, reliability >= 0.97), shuffle with a recorded seed, rent nothing in the first 37% (the benchmark sample), then rent the first offer whose score (GPUs x bf16 TFLOPS per GPU / $ per hour) beats the sample's best. A dud is destroyed and does not reset the benchmark. Full rule: `memory/box-rental-secretary-rule-2026-10-28.md`.
- **2x and 4x GPU boxes**: only when the queue has at least one independent run per GPU, each pinned with `CUDA_VISIBLE_DEVICES`, with cores and RAM per GPU at the single-GPU level and no higher price per GPU-hour.
- **Provisioning probe (user, 2026-10-28)**: run `bash scripts/ops/box_provision_probe.sh PORT root@HOST` once SSH answers, before any job is shipped. It checks the GPUs for foreign load, HF, and box->E: speed. The Drive check needs the Drive token on the box, so the token goes on only after the E: and HF checks pass; then the probe's Drive check runs (pass = both >= 2 MB/s). HF upload speed is not measured yet.
- **Token push commands (user, 2026-10-28)**: once a box is connected, Claude outputs the exact token push commands for the user to run, and never runs them: `scripts/ops/push_gdrive_token_to_boxes.sh PORT HOST` (after the E: and HF checks pass) and `scripts/ops/push_hf_token_to_boxes.sh PORT HOST`, with the box's port and bare host. The probe prints both lines at its end. Claude does not read the token files.
- **10-minute run report (user, 2026-10-28)**: every 10 minutes, `python scripts/ops/run_status_report.py` (`--ssh` for a box). Per run: errors with the last error line, progress with the last log lines, whether checkpoint and replay are saved and matched, and ETA in Athens time.

## Evaluation Protocol & Crash Diagnosis (full rules: `.agents/rules/evaluation_protocol_and_crash_diagnosis.md`)
- **CARLA renders at Low by default, Epic only for routes that segfault at Low (user policy, 2026-10-14)**: training data and Bench2Drive's evaluator use CARLA's default (Epic), our harness runs `-quality-level=Low`, which segfaults Town03 and a few other routes (S-137). Run those routes as a separate `CKE ...:CARLA_QUALITY=Epic` job for every compared arm and report them as their own column; never pair runs that differ in any simulator flag.
- **A route is not "broken" until shown**: check where it crashes (load vs drive), search the E: archive for past completions, vary one factor at a time on one host (plain vs switches, WoR, Low vs Epic, another host), run `run_town_probe.sh` for load-time crashes. Exclusions are per (agent, switches, rendering), never global by default. Report route coverage with every DS.
- **Atari numbers against a paper use the paper's protocol** (EZ-V2 400.1: no sticky actions, up to 30 no-op starts); sticky p=0.25 is a separate robustness column (S-138). State the protocol with every score.
- **Resumable runs are backed up as a pair**: an Atari seed resumes only from `checkpoint_latest.pt` + `replay_latest.npz` (saved together every 2,500 updates). Back up both to E: and HF on a wall-clock schedule (every 30 min), verify sizes against the box (a repeated "WRITE FAILED/MISMATCH" is a failure), give each box its own sync timeout, and do the final save at least 30 min before a destroy deadline (S-140).
- A paid box with no running job is idle: flag it in the 10-minute check and start the next prepared job.
