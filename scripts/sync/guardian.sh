#!/usr/bin/env bash
# usage: guardian.sh <label> <ckpt_json> <total_routes> <cmd...>
# Kept for the older callers (run_leaderboard_official.sh and friends). It was a copy of
# scripts/eval/b2d_guardian.sh without the attempt cap, so a route that kills the server every time
# relaunched forever; it now delegates to that one implementation (MAX_ATTEMPTS, default 25).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../eval/b2d_guardian.sh" "$@"
