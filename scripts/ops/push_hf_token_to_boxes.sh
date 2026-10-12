#!/usr/bin/env bash
# RUN THIS YOURSELF (Claude must not read .env): puts the HF token from .env into /dev/shm/hf_token on each box. Prints only the byte count per box.
# Boxes (2026-10-11): Atari 5950X 35145@ssh7 (seed 2), 5600G 15895@ssh3, 3600 31977@ssh1, 5600G(2) 13001@ssh5; CARLA 4090 EPYC 14463@ssh4, 4090 3900X 12723@ssh4, 3090 5950X 21207@ssh4
cd "$(dirname "$0")/../.." || exit 1
TOK=$(grep -m1 "^HF_TOKEN=" .env | cut -d= -f2- | tr -d '\r"'"'"' ')
for hp in "15895 ssh3.vast.ai" "35145 ssh7.vast.ai" "31977 ssh1.vast.ai" "13001 ssh5.vast.ai" "14463 ssh4.vast.ai" "12723 ssh4.vast.ai" "21207 ssh4.vast.ai"; do
  set -- $hp
  printf '%s' "$TOK" | ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p $1 root@$2 'umask 077; cat > /dev/shm/hf_token; echo "$(hostname): token bytes $(wc -c < /dev/shm/hf_token)"' 2>&1 | grep -v "Welcome\|Have fun\|AI agents\|Warning: Permanently"
done
