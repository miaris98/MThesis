#!/usr/bin/env bash
# Waits for /dev/shm/hf_token, restores GTrXL seed 0 (89,496 env steps + replay) from HF, unpacks the replay and resumes training to 100k.
cd /workspace
until [ -s /dev/shm/hf_token ]; do sleep 10; done
echo "$(date -u) token present, restoring" >> /workspace/resume_s0.log
/venv/main/bin/python /workspace/restore_s0_final.py >> /workspace/resume_s0.log 2>&1 || { echo "$(date -u) RESTORE FAILED" >> /workspace/resume_s0.log; exit 1; }
CK=/workspace/MThesis/results/100k_benchmark/S058_ezv2_match/S058h_gtrxl_100k_s0_uniform/checkpoints
cd $CK && ls -la >> /workspace/resume_s0.log
[ -f replay_latest.npz.gz ] && gzip -dc replay_latest.npz.gz > replay_latest.npz && rm -f replay_latest.npz.gz
[ -f checkpoint_latest.pt ] && [ -f replay_latest.npz ] || { echo "$(date -u) MISSING checkpoint_latest.pt or replay_latest.npz" >> /workspace/resume_s0.log; exit 1; }
echo "$(date -u) launching resume" >> /workspace/resume_s0.log
cd /workspace && bash MThesis/scripts/atari/run_s058h.sh 0 0
