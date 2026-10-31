#!/bin/bash
# Provisioning probe for a new Vast box (user rule, 2026-10-28). Run it once the box answers SSH, before any job or token is shipped.
#   Box fit test: GPUs and their load before we use them, cores, CPU model, RAM, disk, pids limit.
#   HF: DNS, a HEAD request and a 200 MB range download from huggingface.co.
#   Box -> E: : a 200 MB file pulled over SSH to E:\MThesis_EXP\_probe (deleted afterwards).
#   Box -> Google Drive: a 200 MB rclone upload (needs /dev/shm/gdrive_rclone.conf on the box, which the user ships with
#     scripts/ops/push_gdrive_token_to_boxes.sh; without it this step is skipped). The test file is deleted afterwards.
#   Box -> HF upload is not measured here: it needs a scratch repo the user approves. Measure it on the first sha-checked push.
# Pass: E: and Drive each at or above MIN_MBS (2 MB/s). That is what one replay pair (about 3 GB) needs per 30 minutes.
# usage: bash scripts/ops/box_provision_probe.sh PORT root@HOST
set -u
PORT=${1:?port}
BOX=${2:?user@host}
MIN_MBS=${MIN_MBS:-2}
SCRATCH=${PROBE_DIR:-/e/MThesis_EXP/_probe}
SO="-o StrictHostKeyChecking=no -o ConnectTimeout=30 -o BatchMode=yes"
mkdir -p "$SCRATCH"

echo "== box fit test ($BOX port $PORT)"
ssh $SO -p "$PORT" "$BOX" 'bash -s' <<'REMOTE'
set -u
nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu,memory.used --format=csv,noheader
echo "gpu processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) (no process of ours yet: anything listed is a foreign load, destroy the box)"
echo "cores: $(nproc) | cpu: $(awk -F: '/model name/{print $2; exit}' /proc/cpuinfo | sed 's/^ *//')"
free -g | awk '/Mem/{print "ram total " $2 " GB, free " $7 " GB"}'
df -h /workspace | tail -1
echo "pids limit: $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo unknown)"
echo "hf dns: $(getent ahosts huggingface.co | head -1 | awk '{print $1}')"
curl -s -o /dev/null -w "hf head %{http_code} %{time_total}s\n" --max-time 30 https://huggingface.co/api/models/gpt2
curl -s -L -o /dev/null -r 0-209715199 -w "hf download 200 MB: http %{http_code}, %{speed_download} B/s\n" --max-time 300 https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
head -c 209715200 /dev/urandom > /dev/shm/probe_200m.bin
REMOTE

echo "== box -> E: (pull of 200 MB over SSH)"
t0=$(date +%s.%N)
ssh $SO -p "$PORT" "$BOX" 'cat /dev/shm/probe_200m.bin' > "$SCRATCH/probe_200m.bin"
t1=$(date +%s.%N)
rm -f "$SCRATCH/probe_200m.bin"
EMBS=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", 200/(b-a)}')
echo "box -> E: $EMBS MB/s"

echo "== box -> Google Drive (rclone, 200 MB, then deleted)"
DRV=$(ssh $SO -p "$PORT" "$BOX" 'bash -s' <<'REMOTE'
if [ ! -f /dev/shm/gdrive_rclone.conf ]; then echo "skipped (no token on the box)"; exit 0; fi
t0=$(date +%s.%N)
rclone --config /dev/shm/gdrive_rclone.conf --retries 2 copyto /dev/shm/probe_200m.bin gdrive:MThesis_backups/_speedtest/probe_200m.bin
rc=$?
t1=$(date +%s.%N)
rclone --config /dev/shm/gdrive_rclone.conf deletefile gdrive:MThesis_backups/_speedtest/probe_200m.bin >/dev/null 2>&1
echo "rc=$rc $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", 200/(b-a)}') MB/s"
REMOTE
)
ssh $SO -p "$PORT" "$BOX" 'rm -f /dev/shm/probe_200m.bin' || true
echo "box -> Drive: $DRV"

echo "== verdict (pass = E: and Drive at or above $MIN_MBS MB/s)"
DMBS=$(echo "$DRV" | sed -n 's/.*rc=[0-9]* \([0-9.]*\) MB\/s.*/\1/p')
if [ -z "$DMBS" ]; then echo "Drive not measured: a Drive-backed plan is not possible on this box until the token is shipped"; fi
awk -v e="$EMBS" -v d="${DMBS:-0}" -v m="$MIN_MBS" 'BEGIN{ if (e+0 >= m && d+0 >= m) print "PASS"; else print "FAIL: below " m " MB/s (E: " e ", Drive " d ")" }'

# Token push: Claude never runs these (it cannot read the token files). Print them for the user to run, after a PASS on the E:/HF checks.
echo "== token push, run by the user (not by Claude):"
echo "   scripts/ops/push_gdrive_token_to_boxes.sh $PORT ${BOX#*@}"
echo "   scripts/ops/push_hf_token_to_boxes.sh $PORT ${BOX#*@}"
