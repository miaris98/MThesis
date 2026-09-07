# WoR decision-head study — status and next steps

Companion to `challenges/challenges_13_transformer_head_underperformance.md` (sections
13.1–13.22), which carries the full reasoning. This file is the working checklist.

Last updated: 2026-09-07, after the four-town comparison.

---

## Where the result currently stands

**The Qwen transformer head beats the conv head on held-out waypoint loss, and the margin
survives the route draw.**

| | qwen30m | cnn |
|---|---|---|
| held-out loss | **0.5715** | 0.6616 |
| paired difference | **−0.0900** (13.6% relative) | |
| 95% CI on difference | **[−0.1160, −0.0670]** — excludes zero | |
| routes won | **78 / 98** (p = 1.4×10⁻⁹) | |
| generalisation gap | 1.52× | |

Conditions: 4 towns / 656 routes / 114,556 frames, 15 epochs, batch 32, `resnet34`,
`--vision_grid 4`, `--val_split 0.15`, `--split_seed 0`, seed 0.

**Caveat that remains open:** n=1 seed per arm. The paired route bootstrap is the correct
test for this comparison and it is decisive, but across-seed variation on the new dataset
has not been measured. Seed runs were launched and lost when the instance was shut down.

---

## Done

- [x] Diagnosed five defects handicapping the transformer: globally-pooled vision, no
      positional/type embeddings, weight decay on residual gates, shared grad-clip
      threshold, no LR warmup. (13.3–13.6)
- [x] Fixed all five; added `--vision_grid`, `--pool_vision`, `--grad_clip`,
      `--warmup_frac`, `--decay_gates_and_norms`.
- [x] Added a real held-out split (`--val_split`, route-boundary aware) — previously
      `is_best` was selected on *training* loss. (13.8)
- [x] Seed control and reproducibility (`src/training/seeding.py`, `--seed`,
      `--deterministic`).
- [x] Config-stamped checkpoints so `load_wor_model` rebuilds the right geometry. (13.10)
- [x] Nine-run seeded sweep on Town01. (13.13)
- [x] **Retracted** that sweep's headline — checked against across-seed spread, which holds
      the held-out set fixed and so certifies against the smaller variance source. (13.17)
- [x] Built `check_val_noise.py`: route-level bootstrap, paired model-vs-model comparison,
      train/held-out gap.
- [x] Confirmed spatial-vision attribution survives the paired test:
      `cnn` vs `cnn_pooled` = +0.0488, CI [+0.0207, +0.0828]. (13.9, 13.17)
- [x] Found the 5.45× memorisation gap — reversing 13.2's "underfitting" for the fixed
      config. (13.18)
- [x] `download_pdm_lite.py`: disk-aware, smallest-first, extract-and-delete, resume-safe.
- [x] Expanded dataset 1 → 4 towns (129 → 656 routes) in ~6 min at ~85 MB/s.
- [x] k-fold cross-validation over routes (`--num_folds`, `--fold`) — implemented and
      verified, **not yet used** (the dataset expansion made it unnecessary for this run).
- [x] Re-ran the comparison on 4 towns: margin holds at 13.6%, gap collapses to 1.52×.
      (13.20, 13.21)
- [x] Results archived to `E:\MThesis_EXP\multitown\multitown.tar` (998 MB — both
      checkpoints, telemetry, run configs, MLflow store).

---

## Next — Tier 1

- [ ] **Re-run the size curve (10M / 30M / 100M).** "Size doesn't matter" was measured
      inside the memorisation regime, where capacity *couldn't* help. That conclusion is
      now suspect and 100M may win. Cheap and high-information.
- [ ] **More data.** Dominant lever twice over. `python download_pdm_lite.py --towns
      Town04,Town05 --reserve-gb 30` → 47 GB, ~9 min, ~1,840 routes total.
- [ ] **`--vision_grid 8`.** The one mechanism confirmed twice. Grid 4→8 nearly doubles
      per-cell discrimination (1.89 → 2.97). ~3× compute.
- [ ] **Closed-loop CARLA evaluation.** Nothing in this entire group has measured driving.
      A 13.6% L1 margin may not survive the PID controller — the error is dominated by
      longitudinal displacement, which is near-closed-form from the speed scalar, while the
      controller steers off lateral only. (13.16, 13.22)
      - Launch CARLA first: `nohup su carlauser -c '/workspace/carla/CarlaUE4.sh
        -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low' &`
- [ ] **Seed replication** of the four-town comparison (2 more seeds per arm, ~25 min).

## Next — Tier 2

- [ ] **`--grad_clip`.** 93% of `qwen30m` batches were rescaled at the default 5.0 and it
      won anyway. Try 10–20.
- [ ] **Train longer.** 15 epochs was a time budget, not a choice. At 1.52× gap there is
      headroom.
- [ ] **LR sweep.** 3e-4 was inherited from the conv head and never revisited; both the
      schedule and the data have since changed.

## Next — Tier 3 (now measurable — paired CI is ±0.025, so ~4% effects resolve)

- [ ] **Bundle:** QK-norm + 2D axial RoPE over the vision grid + register tokens +
      stochastic depth. Measure as one arm; ablate within only if the bundle clears.
- [ ] **Attention-pooled readout** replacing `tokens[:, 0]`, or a state-initialised policy
      token — the structural half of 13.5, still unaddressed.
- [ ] **Earlier backbone stage** (ResNet stage 3 → 16×16 at stride 16) for higher spatial
      resolution.
- [ ] Muon optimiser — own arm, changes the comparison basis.

Regularisation (dropout/augmentation) has **dropped in priority**: at a 1.52× gap there is
little left to recover.

---

## Reusable tooling built along the way

| Tool | Purpose |
|---|---|
| `check_val_noise.py` | Route bootstrap, paired comparison, train/held-out gap |
| `download_pdm_lite.py` | Disk-aware dataset download |
| `compare_wor_runs.py` | Multi-run table with across-seed spread |
| `run_wor_sweep.sh` | Sequential resumable sweep |
| `sync_experiments.py` | tar-over-ssh pull with live progress |

## Methodological rules earned the hard way

1. Repeating a run at several seeds measures **only what you varied**. It holds the
   held-out set fixed, so it cannot certify a margin against route-draw uncertainty.
2. Compare two models with a **paired** bootstrap over the shared held-out routes. Two
   overlapping absolute CIs is the classic over-conservative error.
3. Routes are the independent unit, not frames — frames within a route are near-duplicates.
4. Check the **train/held-out gap before interpreting any comparison**. Under memorisation
   you are measuring which model overfits more gracefully.
5. A dataset too small to learn from is also too small to compare on — and expanding it was
   cheaper than every model-side fix attempted before it.
