# Shared pieces of the Atari EZ-V2 launchers (scripts/atari/run_b*.sh). Source it, do not run it:
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/ez_train.sh"
#
# The launchers used to copy the trainer environment, the NVIDIA MPS start and the resumable retry loop; this is the one copy.
# Paths come from common.sh (WORKSPACE, MTHESIS_ROOT); PY_MAIN is the trainer venv. Tested by tests/test_scripts_ez_train.py.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${PY_MAIN:=/venv/main/bin/python}"   # the trainer venv of the Vast PyTorch template

# ez_env GPU: the trainer environment on one GPU, from the repo root.
ez_env() {
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MLFLOW_ALLOW_FILE_STORE=true \
         PYTHONPATH=$MTHESIS_ROOT CUDA_VISIBLE_DEVICES=$1
  cd "$MTHESIS_ROOT" || exit 1
}

# ez_mps: NVIDIA MPS, so several runs share a GPU (B64: 0.31 s per update each for three runs under MPS, 0.40 s without).
ez_mps() {
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log
  mkdir -p /tmp/mps_pipe /tmp/mps_log
  pgrep -f nvidia-cuda-mps-control > /dev/null 2>&1 || nvidia-cuda-mps-control -d
}

# ez_train_resumable LABEL OUT LOGDIR FLAGS...: train_ez_offpolicy.py FLAGS on the 100k schedule, training only (a checkpoint every
# 10k env steps, --eval-mode deferred), up to 5 tries. Each try resumes from OUT/LABEL/checkpoints/checkpoint_latest.pt, which is
# saved together with replay_latest.npz every 2,500 updates. Log: LOGDIR/LABEL.log. Returns 0 once a try exits 0.
ez_train_resumable() {
  local label=$1 out=$2 logs=$3; shift 3
  local log=$logs/$label.log ck=$out/$label/checkpoints try res rc
  mkdir -p "$ck" "$logs"
  for try in 1 2 3 4 5; do
    res=""; [ -f "$ck/checkpoint_latest.pt" ] && res="--resume-from $ck/checkpoint_latest.pt"
    echo "$(date -u +%H:%M) start $label try $try ${res:-fresh} on GPU ${CUDA_VISIBLE_DEVICES:-?}" >> "$log"
    "$PY_MAIN" -u atari_qwen/training/train_ez_offpolicy.py "$@" \
      --total-steps 100000 --schedule-steps 100000 --eval-interval 10000 --eval-episodes 10 --eval-mode deferred \
      --run-label "$label" --log-dir "$out" $res >> "$log" 2>&1
    rc=$?; echo "$(date -u +%H:%M) $label exited rc=$rc" >> "$log"
    [ $rc -eq 0 ] && return 0
    sleep "${EZ_RETRY_SLEEP:-60}"
  done
  return 1
}
