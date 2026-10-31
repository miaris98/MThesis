#!/usr/bin/env bash
# CARLA box side (TODO A68 + A69, 2026-10-28): after provisioning,
#   A68: E15 on the 7 routes TF++ wins on b2d20 (2143 2286 2509 2664 3717 24795 26401), one run each, with per-tick telemetry
#        (B2D_TELEMETRY_DIR) and the A44 collision / stand-still clips (on by default in queue_A9c.sh), Low rendering, two lanes;
#   meanwhile venv_tfpp (setup_tfpp_env.sh, Python 3.10 from uv) and the TF++ all-towns weights;
#   A69: TF++ on all 220 routes in our harness, Low rendering, three lanes (every third route each), after A68 has finished.
# Markers: /workspace/a68/A68_DONE, /workspace/a69/TFPP_READY, /workspace/a69/A69_LAUNCHED. Log: /workspace/a68/run.log
# usage: setsid nohup bash scripts/eval/run_a68_a69.sh > /workspace/a68/run.out 2>&1 < /dev/null &
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/common.sh"
cd "$MTHESIS_ROOT"
mkdir -p $WORKSPACE/a68/tele $WORKSPACE/a69 $WORKSPACE/q
LOG=$WORKSPACE/a68/run.log
say() { echo "$(date -u +%H:%M) $*" | tee -a $LOG; }
until grep -q PROVISION_DONE $WORKSPACE/setup_carla.log 2>/dev/null; do
  grep -q CARLA_FAILED $WORKSPACE/logs_prov/status 2>/dev/null && { say "PROVISION FAILED"; exit 1; }
  sleep 60
done
say "provisioned"
until [ -f $CHECKPOINTS/e15/SHA_OK ]; do sleep 30; done   # written from the PC once the uploaded E15 files match E:'s sha256
CK=carla_armE_aug1_hires/model_epoch_015.pth
mkdir -p $CHECKPOINTS/carla_armE_aug1_hires
[ -f $CHECKPOINTS/$CK ] || cp $CHECKPOINTS/e15/* $CHECKPOINTS/carla_armE_aug1_hires/
TELE="B2D_TELEMETRY_DIR=$WORKSPACE/a68/tele"
setsid nohup bash scripts/eval/queue_A9c.sh 2000 0 "CKE:a68_e15_la:$CK:2143,2286,2509,2664:$TELE" > $WORKSPACE/q/a68_la.out 2>&1 < /dev/null &
P1=$!; sleep 25
setsid nohup bash scripts/eval/queue_A9c.sh 2100 0 "CKE:a68_e15_lb:$CK:3717,24795,26401:$TELE" > $WORKSPACE/q/a68_lb.out 2>&1 < /dev/null &
P2=$!
say "A68 lanes launched ($P1 $P2)"

# TF++ environment and weights while A68 drives
( uv python install 3.10 > $WORKSPACE/a69/uv_py.log 2>&1
  PYDIR=$(dirname "$(uv python find 3.10)"); export PATH=$PYDIR:$PATH
  bash scripts/setup/setup_tfpp_env.sh > $WORKSPACE/a69/setup_tfpp.log 2>&1 || { say "TFPP ENV FAILED"; exit 1; }
  curl -sL -o $WORKSPACE/tfpp_pretrained.zip https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip && \
    mkdir -p $WORKSPACE/tfpp_pretrained && unzip -q -o $WORKSPACE/tfpp_pretrained.zip "pretrained_models/all_towns/*" -d $WORKSPACE/tfpp_pretrained && \
    ls -la $WORKSPACE/tfpp_pretrained/pretrained_models/all_towns >> $LOG && touch $WORKSPACE/a69/TFPP_READY && say "TF++ ready" || say "TFPP WEIGHTS FAILED" ) &

wait $P1 $P2
touch $WORKSPACE/a68/A68_DONE; say "A68 lanes finished"
until [ -f $WORKSPACE/a69/TFPP_READY ]; do sleep 60; grep -q "TFPP .*FAILED" $LOG && exit 1; done
IDS=$(python3 -c 'import re,sys; print(" ".join(sorted(set(re.findall(r"<route[^>]*\bid=\"(\d+)\"", open(sys.argv[1]).read())), key=int)))' $GARAGE/Bench2Drive/leaderboard/data/bench2drive220.xml)
lane() { echo $IDS | tr ' ' '\n' | awk -v k=$1 'NR % 3 == k' | paste -sd, -; }
export EVAL_AGENT=$GARAGE/team_code/sensor_agent.py EVAL_AGENT_CONFIG=$WORKSPACE/tfpp_pretrained/pretrained_models/all_towns \
       EVAL_PYTHON=$WORKSPACE/venv_tfpp/bin/python LABEL=a69_tfpp220 CKPT=$CHECKPOINTS/$CK \
       ARMS="a:2000:8000:$(lane 1) b:2100:8100:$(lane 2) c:2200:8200:$(lane 0)"
bash scripts/eval/launch_b2d20.sh run >> $LOG 2>&1
touch $WORKSPACE/a69/A69_LAUNCHED; say "A69 TF++ lanes launched"
