# Live-day tools of 2026-10-10 (copied from the session scratchpad; paths inside still point at that scratchpad, edit `S=`/`SP=` before reuse)

- `sync_live_boxes.sh`: per box pull to a transit dir, copy to E: with size checks, HF push from the box (`ONLY=carla|atari`, `HFATARI=1`, `REPLAY=1` to gzip + back up the Atari replay buffer).
- `fleet5.sh`: the 10-minute fleet status (Atari pace/ETA, CARLA routes done/invalid, GPU, credit).
- `remain.py` / `regen.py` / `startx.sh` / `rebal.sh`: compute the routes still missing per arm from the pulled result files, regenerate balanced lane files, start extra lanes WITHOUT killing running ones (`startx.sh`), or stop and relaunch all lanes (`rebal.sh`).
- `resume_s0.sh` / `restore_s0_final.py`: restore an Atari seed (checkpoint_latest + replay) from the HF relay and resume with `scripts/atari/run_s058h.sh`.
