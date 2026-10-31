---
trigger: always_on
description: Knowledge graph routing rules for code-review-graph (CRG) MCP tools, graphify (graphify-out/), and the external experiments knowledge graph (E:\MThesis_EXP\graphify-out\).
---

# Knowledge Graph Routing: `code-review-graph` vs `graphify`

Two knowledge graphs are maintained for this repository — pick by question shape and do not query both for the same thing:

1. **`code-review-graph` (CRG)**: Symbol-precise. Use for "where is X defined", "what calls Y", blast-radius/impact of a change, and test coverage for a function. SQLite-backed, sub-second incremental updates, auto-refreshes on file edits.
2. **`graphify` (`graphify-out/`)**: Concept-precise. Use for cross-document questions spanning code + `.md` docs/TODOs/rules (e.g. "why was WoR chosen as the baseline", "what is the rationale behind X design decision", "how do the CARLA setup guides relate to the training pipeline"). Rebuild manually with `graphify update .` after code/doc/rule changes.

---

## 1. MCP Tools: `code-review-graph` (When Available)

**Start with the `code-review-graph` tools to narrow scope, then read the source.** The graph is cheaper than scanning files and gives structural context (callers, dependents, test coverage) that file search cannot.

### When to use graph tools FIRST
- **Exploring code**: `semantic_search_nodes_tool` or `query_graph_tool` instead of Grep
- **Understanding impact**: `get_impact_radius_tool` instead of manually tracing imports
- **Code review**: `detect_changes_tool` + `get_review_context_tool` instead of reading entire files
- **Finding relationships**: `query_graph_tool` with `callers_of`/`callees_of`/`imports_of`/`tests_for`
- **Architecture questions**: `get_architecture_overview_tool` + `list_communities_tool`

### Verify in the source
- Narrow scope with the graph, then read the source. Do not change code from graph output alone.
- For any non-trivial change, read the implementation and the relevant tests before concluding.
- Verify the exact source when touching behavior, database logic, migrations, retries, fallbacks, recovery, or compatibility code.
- When the graph and the source disagree, the source wins. The graph may be stale or may not model that relationship.
- An empty graph result can mean "not indexed" or "not statically visible", not "does not exist".

---

## 2. `graphify` (`graphify-out/`)

This project has a graphify knowledge graph at `graphify-out/` with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase or architecture questions, when `graphify-out/graph.json` exists, first query the graph instead of scanning raw files:
  - Windows command: `& "C:\Users\miari\anaconda3\envs\graphtools\python.exe" -m graphify query "<question>"` (or `graphify query "<question>"` if on PATH).
  - Use `graphify explain "<concept>"` for focused component inspection (e.g. `WorldOnRailsPolicy`, `QwenAtariAgent`, `RolloutBuffer`).
  - Use `graphify path "<A>" "<B>"` to trace relationships between two components.
- If `graphify-out/wiki/index.md` exists, navigate it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when `query`/`path`/`explain` do not surface enough context.
- After modifying code files in a session, run `& "C:\Users\miari\anaconda3\envs\graphtools\python.exe" -m graphify update .` to keep the graph current (AST-only, no API cost).

---

## 3. External Experiments Knowledge Graph (`E:\MThesis_EXP`)

The external drive `E:\MThesis_EXP` contains historical checkpoints, Bench2Drive evaluations, MLflow runs, and experiment logs, indexed with graphify at `E:\MThesis_EXP\graphify-out\`.

Rules:
- For questions about past experiments, historical checkpoints, Bench2Drive scores, or external drive logs, first consult `E:\MThesis_EXP\graphify-out\wiki\index.md`.
- To query the external graph:
  `& "C:\Users\miari\anaconda3\envs\graphtools\python.exe" -m graphify query "<question>" --graph "E:\MThesis_EXP\graphify-out\graph.json"`
- Maintenance: When new runs, checkpoints, or evaluations are synced to `E:\MThesis_EXP`, maintain the external graph by running:
  `python scripts/sync/build_external_graph.py`
  or re-exporting the wiki:
  `graphify export wiki --graph "E:\MThesis_EXP\graphify-out\graph.json" --labels "E:\MThesis_EXP\graphify-out\.graphify_labels.json"`
