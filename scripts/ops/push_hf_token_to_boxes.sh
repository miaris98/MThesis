#!/usr/bin/env bash
# RUN THIS YOURSELF (Claude must not read .env): puts the HF token from .env into /dev/shm/hf_token on each box. Prints only the byte count per box.
# Boxes (2026-10-10): Atari s2 13149@ssh9, s3 17183@ssh5, s1 22785@ssh1, s4 11985@ssh5; CARLA C2 25991@ssh5, C1 11083@ssh9, CY 23835@ssh7, CX 15023@ssh8
cd "$(dirname "$0")/../.." || exit 1
TOK=$(grep -m1 "^HF_TOKEN=" .env | cut -d= -f2- | tr -d '\r"'"'"' ')
for hp in "13149 ssh9.vast.ai" "17183 ssh5.vast.ai" "22785 ssh1.vast.ai" "11985 ssh5.vast.ai" "25991 ssh5.vast.ai" "11083 ssh9.vast.ai" "23835 ssh7.vast.ai" "15023 ssh8.vast.ai"; do
  set -- $hp
  printf '%s' "$TOK" | ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p $1 root@$2 'umask 077; cat > /dev/shm/hf_token; echo "$(hostname): token bytes $(wc -c < /dev/shm/hf_token)"' 2>&1 | grep -v "Welcome\|Have fun\|AI agents\|Warning: Permanently"
done
