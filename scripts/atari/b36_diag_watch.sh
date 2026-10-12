#!/usr/bin/env bash
# B36: run the B15/B18 diagnostics (CPU) on every new checkpoint_env*.pt of one run and keep a one-line-per-checkpoint log, so the
# mixer's weights / attention entropy / gate opening can be read while the run trains. Result next to the checkpoint as diag_<name>.json.
# usage: b36_diag_watch.sh RUN_DIR STATES_NPY      (loops until killed; STATES from HF diag/b15_states_512.npy)
RUN=$1; STATES=$2; cd /workspace/MThesis
while true; do
  for ck in $RUN/checkpoints/checkpoint_env*_upd*.pt; do
    [ -f "$ck" ] || continue
    out=${ck%.pt}; out=$(dirname $ck)/diag_$(basename $out).json
    [ -f "$out" ] && continue
    sleep 20   # the trainer writes through a .tmp file and renames, but let the copy settle
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 /venv/main/bin/python scripts/analysis/b15_b18_diagnostics.py \
      --states-file $STATES --out $out "$ck" 2>&1 | grep -a -v Warning | cut -c1-400
    # the mixer weights themselves (norms) next to the attention readout: |Wq| ~6.5 = still at init, ~0 = decayed
    CUDA_VISIBLE_DEVICES="" /venv/main/bin/python - "$ck" <<'PY' 2>/dev/null
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
k = "repr_mixer.blocks.0"
print("   mixer |Wq| %.3f |FFN gate| %.3f |in| %.3f |out| %.3f out.bias %.3f gate bias %.3f" % (
    sd[k + ".q_proj.weight"].norm(), sd[k + ".w_gate.weight"].norm(), sd["repr_mixer.inp.weight"].norm(),
    sd["repr_mixer.out.weight"].norm(), sd["repr_mixer.out.bias"].norm(), sd[k + ".gate1.bg"].mean()), flush=True)
PY
  done
  sleep 120
done
