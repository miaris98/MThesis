---
trigger: always_on
---

# Struggle & Solutions Log Rule

## Auto-Log Resolved Issues

Whenever the agent **fully resolves a bug, error, or configuration problem** in this project, it MUST:

1. Check `struggle-solutions.md` in the project root.
2. Assign the next available `[S-NNN]` ID (find the highest existing one and increment by 1).
3. Append a new entry using this template:

```markdown
---

## [S-NNN] <Short title of the problem>
**Date**: <YYYY-MM-DD>
**File**: `<file path>:<line>` (if applicable)
**Symptom**: <One sentence description of what the user observed>
**Root Cause**: <Why it happened>
**Fix**:
<code block or bullet points of the fix>
**Commit**: <git commit hash if applicable>
**Status**: RESOLVED
```

## Conflict Detection

Before appending, scan the existing entries for any entry whose **Fix** contradicts the new fix. If a contradiction is detected:
- Do NOT append automatically.
- Inform the user: "⚠️ This fix may conflict with [S-NNN]: <summary>. Should I overwrite, append both, or skip?"
- Wait for the user's decision.

## Never Rewrite Existing Entries

Never modify or delete existing `[S-NNN]` entries. Only append new ones.
