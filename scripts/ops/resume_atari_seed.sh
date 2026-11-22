#!/usr/bin/env bash
# usage: resume_atari_seed.sh SEED NAME [GPU]   waits for /dev/shm/hf_token, restores the seed from HF, unzips the replay, resumes training to 100k.
# Log: $WORKSPACE/resume_s<SEED>.log. Training itself logs to $WORKSPACE/S058h_gtrxl_100k_s<SEED>_uniform.log (run_s058h.sh).
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
SEED=$1; NAME=$2; GPU=${3:-0}; cd $WORKSPACE
LOG=$WORKSPACE/resume_s$SEED.log
until [ -s /dev/shm/hf_token ]; do sleep 10; done
echo "$(date -u) token present, restoring seed $SEED from $NAME" >> $LOG
/venv/main/bin/python $WORKSPACE/MThesis/scripts/ops/restore_atari_seed.py $SEED $NAME >> $LOG 2>&1 || { echo "$(date -u) RESTORE FAILED" >> $LOG; exit 1; }
CK=$WORKSPACE/MThesis/results/100k_benchmark/S058_ezv2_match/S058h_gtrxl_100k_s${SEED}_uniform/checkpoints
cd $CK && [ -f replay_latest.npz.gz ] && gzip -dc replay_latest.npz.gz > replay_latest.npz && rm -f replay_latest.npz.gz
[ -f checkpoint_latest.pt ] && [ -f replay_latest.npz ] || { echo "$(date -u) MISSING checkpoint_latest.pt or replay_latest.npz" >> $LOG; exit 1; }
ls -la >> $LOG
echo "$(date -u) launching resume" >> $LOG
cd $WORKSPACE && bash MThesis/scripts/atari/run_s058h.sh $GPU $SEED
