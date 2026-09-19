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
2. **Stage 2 — Hugging Face Second**:
   - Push verified champion checkpoints, evaluation summaries, and gameplay/rollout videos to Hugging Face using credentials in the local `.env` file (`HF_TOKEN`) via existing sync scripts (`scripts/sync/hf_push_checkpoints.py` or `atari_qwen/training/hf_sync.py`).

### 3. Compute Utilization & Parallelism
Before triggering any training or evaluation run:
- **Maximize Hardware Utilization**: Fully leverage available GPU VRAM, CPU cores, and system memory on the target machine (e.g., RTX A4000 16GB, EPYC 64-core).
- **Safe OOM Headroom**: Size batch sizes, rollout steps, and replay buffer capacities to leave a safe ~10–15% VRAM headroom to prevent CUDA Out-Of-Memory (OOM) crashes.
- **Default to Parallel Execution**: Run workloads in parallel by default (e.g., vectorized simulation environments `num_envs=16` or `32`, multi-worker data loaders `num_workers=4` to `8`, or parallel evaluation trials) unless explicitly instructed by the user to run sequentially or single-threaded.
