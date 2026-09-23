---
name: to-claude
description: >-
  Activated by the user typing `to-claude` in their message. Gathers relevant
  context from the current conversation, codebase state, and user intent, then
  generates a focused, self-contained prompt suitable for sending to another
  LLM agent (e.g. Claude, GPT-4). Strips irrelevant history, keeps only the
  active problem and its context, and outputs a clean prompt block the user can
  copy-paste directly.
---

# `to-claude` Skill

## When to Activate
Activate whenever the user's message starts with or contains `to-claude`.

## What to Do

### 1. Identify the Core Problem
Read the user's message after `to-claude` and extract:
- The **specific question or task** they want the external agent to help with.
- Any **error messages** pasted in the message (verbatim).
- The **file(s) and line numbers** currently open or relevant.

### 2. Gather Minimal Relevant Context
Pull **only** the following — do NOT include the full conversation history:
- The error or symptom (exact traceback if present).
- The relevant file snippet (use `view_file` to read ≤60 lines around the error).
- **Knowledge Graph Context (Memory)**: Query the graphify knowledge graph using `& "C:\Users\miari\anaconda3\envs\graphtools\python.exe" -m graphify explain "<symbol>"` or `query "<problem>"`. If the question involves historical experiments, past checkpoints, or Bench2Drive scores, also check or query the external graph at `E:\MThesis_EXP\graphify-out\graph.json` (`--graph "E:\MThesis_EXP\graphify-out\graph.json"`) or `E:\MThesis_EXP\graphify-out\wiki\index.md`.
- The relevant section of `struggle-solutions.md` if the problem matches a known struggle ID.
- The active training command (if training-related).
- Key environment facts: Python env (`carla_py38`), GPU count, CARLA version, Vast.ai.

### 3. Generate the Prompt Block

Output the prompt as a clean markdown code block the user can copy:

```
=========================================================
 CONTEXT FOR EXTERNAL LLM
=========================================================
Project: MThesis — Autonomous Driving PPO/WoR on CARLA & Atari Qwen RL
Repo: github.com/miaris98/MThesis
RULES: Mandatory MLflow tracking + two-stage sync (E:\MThesis_EXP first -> Hugging Face second via .env). Maximize GPU & CPU co-utilization by running workloads in parallel by default with safe 10-15% OOM headroom.
PROGRESSIVE GATING & MEMORY: All experiments must follow the 4-gate progressive scaling protocol (Gate 1: Smoke POC 3k-5k steps / 1 epoch; Gate 2: Convergence Check 20k steps / 5 epochs; Gate 3: Robustness Check 50k steps / 20 epochs; Gate 4: Full Benchmark 100k / 50 epochs). Every gate MUST save complete resumable checkpoints (model, optimizer, scheduler, step/epoch) and subsequent gates MUST resume directly from previous gate checkpoints (--resume_from) to prevent discarding compute or restarting from scratch. Never recommend launching full 100k/50-epoch runs without Gate 1-2 empirical validation.

PROBLEM:
<one-paragraph summary of the exact problem>

KNOWLEDGE GRAPH MEMORY (Graphify Context):
<subgraph or dependency summary of relevant components from graphify>

ERROR (verbatim):
<paste exact error traceback here>

RELEVANT CODE (<filename>:<start_line>-<end_line>):
<paste ≤50 lines of code>

WHAT HAS ALREADY BEEN TRIED:
<bullet list from struggle-solutions.md if relevant, or "Nothing yet">

QUESTION FOR YOU:
<the specific question the user wants answered>
=========================================================
```

### 4. Conflict Check against struggle-solutions.md
Before generating the prompt, scan `struggle-solutions.md` for any entry whose
**Symptom** or **Fix** directly contradicts what the user is about to ask.
- If a contradiction is found: summarize the conflict and **ask the user to clarify** before generating the prompt.
- If no contradiction: generate the prompt immediately.

### 5. After Generating the Prompt
- Ask the user: "Want me to also append this as a new entry to `struggle-solutions.md`?"
- If yes, generate the next available `[S-NNN]` ID and append using the standard template.
