#!/usr/bin/env bash
# RUN THIS YOURSELF (Claude must not read .env): puts the HF token from .env into /dev/shm/hf_token on each box. Prints only the byte count per box.
# usage:  scripts/ops/push_hf_token_to_boxes.sh PORT HOST ...   # e.g. 40728 85.131.126.173 (the bare host; root@ is added here)
# Takes the boxes as arguments only. The old hard-coded list (2026-10-11 boxes, all gone) was removed so no token goes to a stale host.
[ "$#" -ge 2 ] || { echo "usage: $0 PORT HOST [PORT HOST ...]"; exit 1; }
cd "$(dirname "$0")/../.." || exit 1
TOK=$(grep -m1 "^HF_TOKEN=" .env | cut -d= -f2- | tr -d '\r"'"'"' ')
while [ "$#" -ge 2 ]; do
  PORT=$1; HOST=$2; shift 2
  printf '%s' "$TOK" | ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p "$PORT" root@"$HOST" 'umask 077; cat > /dev/shm/hf_token; echo "$(hostname): token bytes $(wc -c < /dev/shm/hf_token)"' 2>&1 | grep -v "Welcome\|Have fun\|AI agents\|Warning: Permanently"
done
