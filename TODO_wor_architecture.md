# WoR decision-head study — status and next steps

Companion to `challenges/challenges_13_transformer_head_underperformance.md` (sections
13.1–13.28), which carries the full reasoning. This file is the working checklist.

Last updated: 2026-09-09, after running the closed-loop CARLA comparison to completion: cnn
vs qwen30m (baseline) vs qwen30m (+geometry loss), 15 paired routes each. See "Where the
result currently stands" below and challenges_13 §13.31.

---

## Where the result currently stands

**The Qwen transformer head beats the conv head on held-out waypoint loss, and the margin
survives both the route draw and seed replication.**

| seed | qwen30m | cnn | paired diff | 95% CI | routes won | relative margin |
|---|---|---|---|---|---|---|
| 0 | 0.5715 | 0.6616 | −0.0900 | [−0.1160, −0.0670] | 78/98 (p=1.4×10⁻⁹) | 13.6% |
| 1 | 0.5896 | 0.6660 | −0.0757 | [−0.1047, −0.0511] | 77/98 | 11.5% |
| 2 | 0.5874 | 0.6583 | −0.0718 | [−0.0954, −0.0491] | 78/98 | 10.8% |

Conditions: 4 towns / 656 routes / 114,556 frames, 15 epochs, batch 32, `resnet34`,
`--vision_grid 4`, `--val_split 0.15`, `--split_seed 0`. Generalisation gap 1.52× at seed 0.

**Seed replication closed (2026-09-08):** every seed individually clears the paired
bootstrap — CI excludes zero all three times, and routes-won is stable at 77-78/98. qwen's
own across-seed spread (0.5715–0.5896, 1.8% wide) sits far inside the qwen-cnn gap (0.069 at
the closest pair, 6-8× the within-arch noise). Not a seed-0 fluke.

**`--vision_grid 8` tested, did not help (2026-09-08, seed 0 only):** qwen 0.5835 held-out
(vs 0.5715 at grid4 — 2.1% *worse*), cnn 0.6613 (vs 0.6616 — flat). Paired diff −0.0782, CI
[−0.1020, −0.0571], 73/98 routes: the margin is unchanged within normal seed noise (compare
the 10.8–13.6% range above), not larger as hypothesised. qwen also costs ~13% more per epoch
at grid8 (90s vs 80s, from the longer 68-token sequence); cnn's cost is flat (conv head, not
attention, so more spatial cells is cheap). Grid resolution beyond 4×4 is not currently a
productive lever here — deprioritise below the remaining Tier-1 items unless a size-curve or
more-data result changes the picture.

**Size curve is flat — capacity is not the constraint (2026-09-08, seed 0):** qwen10m
(10.8M) 0.5913, qwen30m (27.5M) 0.5715/0.5896/0.5874, qwen100m (106.5M) 0.5895. All three
sizes land in a 3.5% band against qwen30m's own 1.8% seed spread — a **tenfold parameter
range produces no resolvable difference**, and this is now out of the memorisation regime on
5× the data, so it is not 13.2's underfitting story either. qwen100m converges *later* (0.6535
at epoch 7 vs qwen10m's 0.6282) and recovers the whole deficit in the cosine tail, so a size
curve read mid-schedule reports the schedule rather than the capacity. It costs 128s/epoch
against qwen10m's 95s for a number 0.3% apart.

**What that leaves.** Three architectural hypotheses have now been closed by measurement —
grid resolution (13.24), capacity (13.25), and positional embeddings (already present since
13.6, see 13.27). The remaining levers are not capacity or connectivity but *what the head is
supervised on* and *what it is allowed to express*. Hence the two items now at the top of
Tier 1.

**Geometry-loss A/B: heading error down 90%+, val_loss unmoved (2026-09-08, seeds 0/1/2):**
`--heading_loss_weight 0.5 --curvature_loss_weight 0.2` against a freshly matched baseline
(same code, same 656-route data, same seeds, weights zeroed — the original baseline's
checkpoints no longer existed to compare against directly, since only lightweight artifacts
are kept off-instance; see the note below). Paired bootstrap on `val_loss` at all three seeds:
**NOT SUPPORTED** — every CI spans zero, routes-won 46/98, 48/98, 43/98, indistinguishable
from a coin flip. The loss did not cost anything on the metric it was required not to move.
On the metric it was built to move, `val_wp_heading_err` fell from a baseline mean of 0.043 to
a geometry-loss mean of 0.0034 — a 92% reduction, same direction and same order of magnitude
at every seed, not a bootstrap-shaky effect. `val_wp_curvature_err` improved a smaller ~2.5%.
**But `val_wp_lateral_error_m` — the one quantity `PIDController.control_from_waypoints`
actually reads — was flat within noise** (baseline mean 0.0517, geometry-loss mean 0.0522).
So the loss visibly straightens the predicted path's *direction* between waypoints without
moving the *position* of any single waypoint measurably. Whether that helps the vehicle drive
is not something held-out loss of any kind can answer — it depends on how the PID's lookahead
point selection responds to a smoother path, which is exactly the question closed-loop eval
exists to answer. This moves closed-loop CARLA evaluation from "the remaining gate" to
"the only thing left that can adjudicate this."

**Closed-loop run complete (2026-09-09): the two-improvement bundle wins, neither
improvement alone is certified.** 15 paired Town01 routes, same traffic seed, all three
checkpoints retrained fresh (4 towns / 656 routes, matching every prior result exactly).
Point estimates: cnn mean DS 0.244, qwen30m baseline 0.278, qwen30m+geometry 0.327. Paired
bootstrap: cnn-vs-baseline **NOT SUPPORTED** (CI [-0.100, +0.032]), baseline-vs-geometry
**NOT SUPPORTED** (CI [-0.145, +0.064]), but cnn-vs-geometry **clears** on both driving score
(CI [-0.148, -0.012]) and route completion (CI [-0.068, -0.004]). At 15 routes each single
step is indistinguishable from route-draw noise; only the accumulated change across both
improvements survives the same test. Full writeup: challenges_13 §13.31.

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
- [x] Seed replication of the four-town comparison (seeds 1, 2) — margin confirmed at every
      seed via paired bootstrap, CI excludes zero all three times. Lightweight artifacts
      (telemetry/config/logs, not checkpoints) kept in `results/csvs/`.
- [x] Tested `--vision_grid 8` (seed 0) — no improvement for either arch, qwen slightly
      worse and costs more compute; deprioritised.
- [x] Re-ran the size curve (10M/30M/100M) on 656 routes — flat. Retires the "100M may win
      now" hypothesis. (13.25)
- [x] Measured where the objective actually spends itself: 73.5% of held-out loss is
      longitudinal, which the PID never reads for steering. Added `--heading_loss_weight`
      and `--curvature_loss_weight` (scale-free, default 0, `val_loss` deliberately
      unchanged so the whole result series stays comparable) + 6 tests. (13.26)
- [x] Ran the geometry-loss A/B: `qwen30m` seeds 0/1/2, `--heading_loss_weight 0.5
      --curvature_loss_weight 0.2` vs a freshly matched baseline. `val_loss` unchanged (all 3
      paired bootstraps NOT SUPPORTED), `val_wp_heading_err` down 92%, `val_wp_curvature_err`
      down ~2.5%, `val_wp_lateral_error_m` flat. Free to keep; closed-loop eval is now the only
      way to know if it matters. (13.29)
- [x] Audited whether the `route` input leaks the `waypoints` target. It does not: targets
      come from `ego_matrix`, route from the planner's plan, and the entire 5-waypoint
      horizon (mean 5.27 m) sits inside the *first* route segment (route points at
      2.6/8.6/15.3/20.8 m). A no-vision (route+speed+command) MLP floor of 0.7558 reframes
      the result — the transformer extracts 1.87x the visual signal the conv head does,
      measured against what vision is worth rather than against zero. (13.30)
- [x] Built closed-loop CARLA evaluation end to end and verified it against a live server:
      `src/eval/driving_metrics.py` (Leaderboard 1.0 Driving Score, no `carla` import, 16
      unit tests), `eval_wor_closed_loop.py` (deterministic rollout — seeded routes AND
      seeded traffic, so checkpoints are comparable route-by-route), `compare_closed_loop.py`
      (paired bootstrap), `src/eval/rollout_video.py` (optional chase-cam recording).
      Found and fixed three real bugs building it: a speed-units mismatch present since
      `wor_policy.py`'s `act()` was written (network trained on m/s, PID needs km/h, the two
      existing callers each passed the wrong one to one consumer — see 11.6); a collision
      double-counting artifact (one continuous 30 s wedge against a van logged six
      infractions instead of one, see 11.7); and a third variant of the Vulkan driver gap,
      where the ICD manifest pointed at a library file that was simply absent rather than
      mismatched — `vulkaninfo` cannot tell the two apart from its error message alone
      (`carla_vulkan_driver_troubleshooting_guide.md`, 1.6a).
- [x] Ran all three checkpoints concurrently for the first time, hit a traffic-manager port
      collision invisible from the CARLA server's own logs (`get_trafficmanager()` binds a
      fixed port regardless of `--port`), fixed with an explicit `--tm_port` per arm. See
      11.8 for the two failure modes it produced (a catchable `RuntimeError` on one arm, an
      uncatchable C++ `std::terminate` on another, depending on timing).
- [x] **Completed the closed-loop comparison: 15 paired routes, cnn vs qwen30m baseline vs
      qwen30m+geometry-loss.** Only the two-improvement bundle (cnn vs geometry-loss) clears
      a paired bootstrap; neither single step does at this sample size. Recorded and pulled
      chase-cam video for the first 2 routes of each arm. (13.31)

---

See also [TODO_optuna_and_resources.md](TODO_optuna_and_resources.md) for the Optuna
hyperparameter-search plan (the structural fix for 13.6's shared-defaults problem) and an
assessment of nine external CARLA datasets/models.

## Next — Tier 1

- [ ] **More data.** Dominant lever twice over. `python download_pdm_lite.py --towns
      Town04,Town05 --reserve-gb 30` → 47 GB, ~9 min, ~1,840 routes total.
- [ ] **Widen the closed-loop sample.** 15 routes was enough to certify the two-improvement
      bundle (cnn vs qwen30m+geometry) but not either single step (13.31) — the CIs on those
      two comparisons ([-0.100, +0.032] and [-0.145, +0.064]) are wide enough that more
      routes could plausibly resolve one or both. `run_closed_loop_arms.sh` already runs all
      three arms concurrently (`--tm_port` fix, 11.8); raising `--routes` past 15 is the only
      change needed, budget ~2.5 min/route/arm when solo, less under concurrency.
- [ ] **Attribute the two-improvement bundle.** 13.31 shows cnn-vs-geometry clears but
      neither cnn-vs-baseline nor baseline-vs-geometry does alone — that's a sample-size
      argument, not evidence the architecture change contributes nothing on its own. A wider
      route sample (above) is the direct way to find out whether baseline-vs-geometry
      resolves on its own with more routes, which is the more informative of the two
      open comparisons since architecture-only was already the certified open-loop result.

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
6. A point-estimate delta and a paired-bootstrap verdict can disagree, and the bootstrap
   wins: 13.29's raw numbers looked like a small, consistent regression across all three
   seeds; the paired test showed every one of those seeds was a coin flip. Trust the CI
   over the sign of three numbers, even when the sign agrees three times.
7. Keeping only lightweight artifacts off-instance is cheap until a later comparison needs
   the checkpoint itself — 13.29 could not be bootstrapped against the original baseline
   because those `.pth` files were never kept, and retraining a matched baseline cost real
   (if small) time. If a follow-up comparison is foreseeable, keep the checkpoint until it
   isn't.
