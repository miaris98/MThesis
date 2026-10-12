---
trigger: always_on
description: Consult the graphify knowledge graph at graphify-out/ for codebase and architecture questions.
---

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- For codebase or architecture questions, when `graphify-out/graph.json` exists, first query the graph instead of scanning raw files.
- Command execution on this Windows system: Use `& "C:\Users\miari\anaconda3\envs\graphtools\python.exe" -m graphify query "<question>"`, or `explain "<concept>"`, or `path "<A>" "<B>"`.
- Use `graphify explain "<concept>"` for focused component inspection (e.g. `WorldOnRailsPolicy`, `QwenAtariAgent`, `RolloutBuffer`).
- Use `graphify path "<A>" "<B>"` to trace interactions between two components.
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code files in this session, run `& "C:\Users\miari\anaconda3\envs\graphtools\python.exe" -m graphify update .` to keep the graph current (AST-only, no API cost).

### External Experiments Knowledge Graph (E:\MThesis_EXP)

The external drive `E:\MThesis_EXP` contains historical checkpoints, Bench2Drive evaluations, MLflow runs, and experiment logs, indexed with graphify at `E:\MThesis_EXP\graphify-out\`.

Rules:
- For questions about past experiments, historical checkpoints, Bench2Drive scores, or external drive logs, first consult `E:\MThesis_EXP\graphify-out\wiki\index.md`.
- To query the external graph: `& "C:\Users\miari\anaconda3\envs\graphtools\python.exe" -m graphify query "<question>" --graph "E:\MThesis_EXP\graphify-out\graph.json"`.
- When new runs or checkpoints are synced to `E:\MThesis_EXP`, run `python scratch/build_external_graph.py` or re-export the wiki to update the index.
