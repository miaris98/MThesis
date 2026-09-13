# Continuation handoff — WoR v2 retrain (written 2026-09-13, ~18:00)

Read this, then delete it (repo convention from the previous handoff).

---

## 1. What is running RIGHT NOW

Two training arms on the vast.ai box, launched ~13:46, detached via `setsid` (they survive SSH
drops and the end of any Claude session — they do **not** depend on a conversation being open):

```
ssh -p 26597 root@ssh3.vast.ai          # RTX 3080 Ti, 12GB, 16 vCPU, 62GB RAM
/workspace/train_v2_cnn.log             # arm 1: cnn
/workspace/train_v2_qwen.log            # arm 2: qwen30m+geom
/workspace/checkpoints/wor_v2/cnn_s0/
/workspace/checkpoints/wor_v2/qwen30m_geom_s0/
```

Launched by `launch_retrain_v2.sh`. 50 epochs each, `--batch_size 32` (pinned, do not change).

**Measured per-epoch cost (from `wor_training_telemetry.csv`, not estimates):**

| arm | train | val | wall/epoch | 50 epochs |
|---|---|---|---|---|
| cnn | 1048 s | 159 s | **1208 s (20.1 min)** | ~16.8 h |
| qwen30m+geom | 1690 s | 188 s | **1880 s (31.3 min)** | ~26 h at current rate |

Qwen accelerates once cnn finishes (see §5), so realistic all-done is **~22 h from 13:46**.

Progress as of writing: cnn epoch 5, qwen epoch 3. Both healthy, GPU pinned 98-99%.

**Health check one-liner:**
```bash
ssh -p 26597 root@ssh3.vast.ai "pgrep -f train_wor.py | wc -l; \
  grep -cE 'Epoch 0' /workspace/train_v2_cnn.log /workspace/train_v2_qwen.log"
```

Val ADE is improving normally on both (cnn 0.428→0.369 over epochs 2-5; qwen ~0.42).

---

## 2. What v2 changed vs the old baselines (cnn 59.14 / qwen 56.86 on Bench2Drive)

Five changes, shipped together **deliberately un-attributable** (user's explicit call: speed over
ablation). If attribution is needed later, the cheap follow-ups are `--route_overlay 0` and
`--target_speed_loss_weight 0`; the camera fix is a bug fix and would not be reverted.

1. **Camera parity (the big one).** Dataset was rendered at `x=-1.5, z=2.0, fov=110, 1024x512`;
   the eval agent was requesting `x=+1.3, z=1.3, fov=100, 256x256`. Every closed-loop number in
   this project so far was produced through a camera the network never trained on. Plus a
   BGRA→BGR alpha-drop bug in the eval path. Now one definition in `src/config/camera.py`, used
   by both the dataset and the agent.
2. **Aspect-correct input.** 192x512 with `crop_bottom_frac 0.25` (bonnet removed, TF++'s crop),
   replacing a 2.7x horizontal squash into 256x256.
3. **CARLA-pretrained backbone.** `regnety_032` lifted from the released TransFuser++ checkpoint
   (`/workspace/tfpp_pretrained/pretrained_models/all_towns/model_0030_0.pth`), loads
   **492/492 tensors clean**. `pretrained=False` is load-bearing — there is deliberately no
   ImageNet fallback (user constraint: CARLA-pretrained vision ONLY).
4. **Target-speed head.** 8-bin two-hot with an exact-zero class. Bins match TF++ exactly.
5. **Route overlay.** Planned route projected into the image like a reversing camera's guide
   lines. Drawn pre-crop, in source pixel geometry.

Also: `route_points 4 → 20` (matches TF++). **Caveat worth stating in the thesis:** the full
20-point route correlates ~0.84 with the lateral target, so it inflates absolute numbers. It
affects both arms equally, so the head-vs-head comparison stays fair.

**Why the cnn arm is being retrained too:** the old 59.14 was measured through the broken camera.
Retraining both under the identical fixed pipeline is what makes "transformer vs conv head" a
clean ablation rather than a measurement of bug fixes.

---

## 3. Uncommitted work in progress: feature caching (NOT deployed, NOT tested)

The single biggest remaining speedup. Local edits are on disk, uncommitted, **not yet on the box**.

**Why it works:** the backbone is frozen, `CarlaPretrainedEncoder.train()` forces eval mode so
BatchNorm never drifts, and there is **zero RGB augmentation** anywhere in the pipeline. So
`encoder(frame)` is bit-identical on epoch 1 and epoch 50. Measured: the backbone is **90.9% of a
cnn step and 50.5% of a qwen step**. Both arms recompute the same features from the same frames
independently — so it's being computed redundantly twice per step, 50 times over.

**Files changed (all additive, default-off, existing behaviour untouched):**
- `build_feature_cache.py` — NEW. One-time pass, resumable, atomic temp+rename writes.
- `src/training/wor_dataset.py` — `feature_cache_tag` param, `feature_cache_path()` module fn,
  `pixel_cache_tag()` made public, `_load_features()`, `__getitem__` returns either `rgb` OR
  `vision_features` (never both — keeps batch collation schema consistent).
- `src/training/wor_eval.py` — new `to_device_batch_with_features()` (the old 6-tuple
  `to_device_batch` is left untouched for `check_val_noise.py`).
- `src/training/wor_trainer.py`, `wor_dataloaders.py` — plumb `feature_cache_tag` through.
- `src/models/world_on_rails/wor_policy.py`, `qwen_wor_policy.py` — `forward()` gains
  `vision_features=None`; when given, skips `self.encoder(rgb)` and `rgb` may be None.
  `B`/device now derived from `feats`, not `rgb`.

**Remaining steps:** deploy → run the test suite on the box → validate cached-vs-live loss on a
small slice (`--limit_frames`) → measure real throughput → THEN decide whether to swap the live
run. **Do not kill the running jobs for this without the user's explicit go-ahead.**

**Disk math (checked):** feature is 1512x6x16 fp16 = 283.5 KB/frame ≈ the pixel cache's 288 KB,
so ~144 GB for 520k frames. 399 GB free. Note the pixel cache (~145 GB, already written) becomes
**dead weight once features are cached** — training never decodes a JPEG again — so it can be
deleted to net out roughly even.

Because feature size ≈ pixel size and current data wait is only **0.16%** of step time, disk I/O
should NOT become the new bottleneck.

---

## 4. Bugs found and fixed this session (all verified on the box)

- `--help` crashed: bare `73%` in a help string parsed as a format spec. **Pre-existing at HEAD**,
  not a regression.
- `--max_batches` didn't cap validation — the code contradicted its own comment. A "smoke test"
  ran 100 train batches then all ~2,070 val batches, making timings meaningless.
- `run_step` crashed on a `__new__`-built agent (my regression). Fixed with class-level defaults
  on `WorldOnRailsAgent`.
- Stale `256x256` assumptions in `train_wor.py`: banner printed a feature grid the run never used
  (`8x8` vs the real `6x16`), and `--auto_batch_size` probed at the wrong resolution.
- I pushed `wor_policy.py` and `wor_dataset.py` over the repo's 500-line limit; fixed by
  extracting `vision_encoder.py` and `wor_dataloaders.py` with re-exports (no caller changed).
  That refactor dropped an `os` import and crashed a run — caught it, then ran pyflakes across all
  touched modules.

**Test state on the box: 137 passed.** Only failure is the pre-existing `test_line_counts` (5
files, all untouched by me). Five CARLA-importing test files **segfault** because the vendored
CARLA egg is py3.7 and the venv is py3.12 — environment mismatch, unrelated; exclude them:
```
--ignore=tests/test_driving_state.py --ignore=tests/test_reward.py \
--ignore=tests/test_reward_registry.py --ignore=tests/test_vector_env.py \
--ignore=tests/test_world_on_rails.py
```

---

## 5. Gotchas that cost time this session

- **`scp` silently truncates on this link.** Pulled `frozen_backbone.pth` twice and got 40 MB then
  52 MB of a 72 MB file, both with exit 0. **Use tar-over-ssh and verify with sha256sum.**
- **Background SSH monitors die with exit 255** (dropped connection). This is NOT a training
  failure — always re-check `pgrep -f train_wor.py` before believing anything broke.
- Editing `.py` files on the box does **not** affect the running jobs — Python already imported
  them. (Editing a running *bash* script DOES corrupt it — bash reads by byte offset.)
- `pkill -f <pattern>` over SSH can match your own SSH command string and kill your session.
- MLflow auto-launches a file-backed server; the whole block is try/except so it can't kill a run.

---

## 6. Standing user constraints

- `--batch_size 32` is **pinned** across the project. Do not change.
- CARLA-pretrained vision **only** — never ImageNet.
- **No new sensors** (no LiDAR, no BEV input) unless derived from the RGB the model already gets.
  Depth/BEV/boxes are legitimate *supervision targets*, not inputs. Pseudo-LiDAR from depth is OK.
- Run autonomously, report at milestones; fix and relaunch breakages without asking.
- Commit only when explicitly asked.
- Back up progress to `E:\MThesis_EXP\` (external drive) or HuggingFace.

**Backed up so far:** `E:\MThesis_EXP\checkpoints\wor_v2\{cnn_s0,qwen30m_geom_s0}\` —
best/latest/frozen_backbone/run_config/telemetry for epochs 1-2, all sha256-verified.
**Re-pull periodically**; `/workspace` does not survive instance destruction.

---

## 7. Not yet built / deferred

- **TF++'s extra losses are INERT for us as currently architected — do not build them blindly.**
  TF++ runs 10 active loss terms (`loss_depth 0.1`, `loss_semantic 0.1`, `loss_bev_semantic 0.1`,
  `loss_center_heatmap/wh/offset/yaw_class/yaw_res 0.1`, `loss_target_speed 0.1`,
  `loss_checkpoint 0.1`; note `loss_wp` is **0.0**, disabled). Eight of those are *perception*
  losses whose job is to shape the **encoder**. Our encoder is frozen: `requires_grad=False` on
  every parameter, and `CarlaPretrainedEncoder.forward_pyramid` runs it inside
  `with torch.no_grad()`. **Gradients from an aux head cannot reach it.** Adding a depth or
  semantic head would train only that decoder — a dead-end branch off the frozen features that
  never feeds the policy — and would change driving behaviour by exactly nothing.
  We already *inherit* that supervision: the TF++ weights we load were trained with those losses.
  The two TF++ losses that act on a trainable path are `loss_target_speed` (we have it) and
  `loss_checkpoint` ≈ our waypoint loss (we have it).
  Extra losses could only help via: **(a)** unfreezing the backbone — but that contradicts the
  standing "vision training is out of scope" constraint, kills feature caching, and is far
  slower; or **(b)** attaching aux heads to the **trainable trunk** (make the qwen trunk itself
  predict depth/BEV as multi-task regularisation) rather than to the frozen feature map. (b) is
  legitimate and unexplored, but it is a research bet, not a known win.
- **TF++ uses `use_grad_clip: 0`** (no clipping at all) with `lr 3e-4`, `batch_size 16`,
  `epochs 31`. We clip at 5.0 and **qwen is clipping 100% of batches** (grad norm ~17-22), so
  every qwen gradient is scaled to ~29% — while cnn's clip rate is already falling naturally
  (13662 → 9351 over epochs 2-5). This asymmetry belongs to the threshold, not the architecture.
  Strong candidate for the next iteration.
- **TF++ augments**: `augment_percentage 0.5`, camera rotation ±5°, translation ±1 m, colour jitter.
  PDM-Lite ships an `rgb_augmented` modality we are not using. Note this would **break feature
  caching** (augmentation makes the encoder output non-deterministic per epoch).
- After training: re-run the 112-route official Leaderboard eval + the 38-route Bench2Drive pass.
  Harness exists: `run_leaderboard_official.sh`, `run_leaderboard_resilient.sh`,
  `check_leaderboard_results.py`, `merge_leaderboard_results.py`, `scenario_breakdown.py`.

---

## 8. THE OPEN DECISION: 24 h is not viable — how to get faster

User's position: the current ~22 h remaining is too slow. Ranked by (speedup x confidence):

1. **Feature caching — the real lever.** Code is committed and deployed to the box; a 256-frame
   validation pass was about to run when the session ended. **Nothing is validated yet.**
   Expected: cnn ~4-6 min/epoch (becomes disk-bound), qwen ~8-15 min/epoch. Plausibly turns 50
   epochs into ~8-11 h, or ~5 h at 30 epochs. **Unmeasured — do not quote these as facts.**
2. **Cut epochs.** 50 epochs x 442k frames = 22.1M sample presentations; the old baseline that
   scored 59.14/56.86 saw 50 x 52.8k = **2.64M**. We are already doing **8.4x** its gradient
   steps. 25-30 epochs would still be 4-5x, and roughly halves wall time. Note: the LR schedule
   is cosine over `--epochs`, so **early-stopping a 50-epoch run is NOT equivalent** — the LR
   never anneals. To get the benefit properly this requires a relaunch with `--epochs 30`.
3. **Validate every N epochs.** Measured cost: 159 s/epoch cnn (13.2%), 188 s/epoch qwen (10%).
   TF++ uses `val_every: 5`. Caveat: `best_model.pth` is selected on `val_loss`, so this
   coarsens which checkpoint ships.
4. `torch.compile` on the trunk (~15-25%, untested; trainer already has a `compile_model` param).

**Do NOT "run the arms sequentially instead of concurrently"** (a suggestion the user received
from Gemini). Concurrent aggregate is 421 + 260 = 681 samples/s; the best any single arm reaches
alone is ~450. Sequential would take **~32 h instead of ~22 h**. Concurrency is already correct.

Suggested plan: validate the cache on a small slice -> measure real throughput -> then kill and
relaunch both arms with caching + `--epochs 30`. Killing costs the ~5 cnn / ~3 qwen epochs done
so far (~2.7 h). **Get explicit go-ahead before killing the live run.**

---

## 9. Suggested first message for the new conversation

> Read `continuation_prompt.md` (committed at 2d1078b), check both training arms are still alive,
> then pick up §8: validate the feature-caching code on a small slice, measure the real speedup,
> and tell me whether to relaunch with caching + fewer epochs before continuing the current run.
