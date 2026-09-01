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
- The relevant section of `struggle-solutions.md` if the problem matches a known struggle ID.
- The active training command (if training-related).
- Key environment facts: Python env (`carla_py38`), GPU count, CARLA version, Vast.ai.

### 3. Generate the Prompt Block

Output the prompt as a clean markdown code block the user can copy:

```
=========================================================
 CONTEXT FOR EXTERNAL LLM
=========================================================
Project: MThesis — Autonomous Driving PPO/WoR on CARLA (Vast.ai GPU instance)
Environment: Ubuntu 20.04, Python 3.8 (carla_py38 conda), PyTorch 1.13, CARLA 0.9.13
Repo: github.com/miaris98/MThesis  branch: fix/reward-state-plumbing

PROBLEM:
<one-paragraph summary of the exact problem>

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
