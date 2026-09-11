#!/usr/bin/env bash
# Launches TCP's remaining Phase 4 routes as 3 parallel chunks, once cnn and qwen
# finished and freed up 2 of the box's 3 concurrent CARLA slots (ports 2000/8000
# and 2010/8010 - already validated concurrent-safe, that's how cnn/qwen/tcp ran
# together in the first place). Each chunk writes into the SAME tcp_full51_perroute
# dir via run_phase4_chunk.sh; since the 3 chunks are given disjoint route-id lists,
# there's no file collision. Only after all 3 `wait` returns do we do the one real
# merge + echo PHASE4_ARM_DONE - matching run_phase4_arm.sh's ending exactly, so the
# existing "grep PHASE4_ARM_DONE" monitor fires once, at genuine full completion,
# not once per chunk.
#
# usage: run_phase4_tcp_parallel.sh <chunk1_ids_csv> <chunk2_ids_csv> <chunk3_ids_csv>
set -x
LABEL="tcp_full51"
CKPT="/workspace/checkpoints/tcp/tcp_b2d.ckpt"
OUT=/workspace/bench2drive_out
PERROUTE_DIR="${OUT}/${LABEL}_perroute"
EXPECTED=38

bash /workspace/MThesis/run_phase4_chunk.sh "$LABEL" tcp "$CKPT" - 2020 8020 0 "$1" &
PID1=$!
bash /workspace/MThesis/run_phase4_chunk.sh "$LABEL" tcp "$CKPT" - 2000 8000 0 "$2" &
PID2=$!
bash /workspace/MThesis/run_phase4_chunk.sh "$LABEL" tcp "$CKPT" - 2010 8010 0 "$3" &
PID3=$!

wait $PID1 $PID2 $PID3

python3 -c "
import json, glob
records = []
for path in sorted(glob.glob('${PERROUTE_DIR}/*.json')):
    try:
        d = json.load(open(path))
        records.extend(d['_checkpoint']['records'])
    except Exception:
        pass
out = {'_checkpoint': {'records': records, 'progress': [len(records), $EXPECTED]}}
json.dump(out, open('${OUT}/${LABEL}.json', 'w'), indent=2)
print('MERGED', len(records), 'records into ${OUT}/${LABEL}.json')
"

final=$(python3 -c "
import json
try:
    d = json.load(open('${OUT}/${LABEL}.json'))
    print(len(d['_checkpoint']['records']))
except Exception:
    print(0)
")
echo "PHASE4_ARM_DONE label=${LABEL} routes_recorded=${final}/${EXPECTED}"
