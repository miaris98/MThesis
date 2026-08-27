# Preserve MLflow and Cloudflare Tunnel Logic in Supervisor Scripts

## Protected Code Sections
In [run_multi_carla_training.sh](file:///c:/Users/miari/Desktop/MThesis/run_multi_carla_training.sh) (and any derived training supervisor scripts), **DO NOT MODIFY, REFACTOR, SIMPLIFY, OR REMOVE** the following sections:

1. **MLflow UI Server Initialization**:
   - Port auto-detection (`find_mlflow_port`).
   - Isolated `tmux` session (`mlflow_server`) management.
   - Server health checks and persistent background daemon execution.

2. **Cloudflare Tunnel & Link Resolution**:
   - `cloudflared` binary detection, auto-installation, and validation (`is_valid_cloudflared`).
   - Log parsing and regex URL extraction (`extract_cf_url`).
   - Tunnel session management in tmux (`mlflow_tunnel`).
   - Exponential backoff retry logic for 429 rate-limits.
   - Cached URL reuse (`/tmp/mlflow_cf_url`) and public DNS reachability checks.
   - The MLflow dashboard banner display (`link to mlflow : <URL>`).

## Strict Constraint
- **Do not touch, edit, replace, or remove** this logic during code edits, cleanups, refactors, or optimizations.
- **Explicit Prompt & Extra Approval Required**: Even if a request involves MLflow or Cloudflare changes, **you MUST explicitly ask for and obtain one extra confirmation/approval from the user** before making any modifications to this protected section.
