# TODO — Optuna hyperparameter search, and external resources assessed

Companion to [TODO_wor_architecture.md](TODO_wor_architecture.md) and
`challenges/challenges_13_transformer_head_underperformance.md`.

Created 2026-09-07.

---

# Part 1 — Optuna

## Why this is the principled fix for Challenge 13.6, not just tuning

Challenge Group 13.6 documented three optimiser settings — grad clip 5.0, no warmup, weight
decay on residual gates — that were "set once, against the first head that was built, and
then inherited unchanged by a head with 59 times the parameters." Cross-cutting theme: *a
shared default is not a fair comparison.*

Manually fixing those three is patching symptoms. The structural fix is to **give each
architecture its own hyperparameter search with an equal budget**, so the comparison is
best-tuned-vs-best-tuned instead of best-vs-whatever-the-other-model-liked. That is what
Optuna buys here, and it is a stronger claim for the thesis than any single tuned number.

Evidence this still matters: in the four-town run, `--grad_clip 5.0` rescaled **1,220 of
1,311** batches for `qwen30m` (93%) and it won anyway. The transformer is being throttled by
a threshold chosen for the conv head, right now, in the run that produced the headline.

## The hard part: the objective is noisy, and Optuna will happily chase noise

This is the thing to get right, and it is where a naive integration would quietly produce a
worthless result. From 13.17 and 13.20:

- The paired 95% CI on a model-vs-model difference is about **±0.025** at 98 held-out routes.
- A single-model held-out loss carries a 95% CI of width **0.155**.

Optuna run for 100 trials against a metric with that much noise will "find" a configuration
better than the rest **by selection on noise**, and report an optimistic number. Three
defences, in order of importance:

1. **A test split that selection never touches.** Partition routes three ways —
   train / val / test. Optuna optimises val; the winning configuration is re-scored once on
   test and *that* is the reported number. Without this, the tuned score is not an estimate
   of anything.
2. **Reduce objective variance with k-fold.** `--num_folds` / `--fold` already exist
   (committed, verified, unused so far). A 3-fold objective costs 3x per trial and cuts the
   objective's standard error by 1.7x. Prefer fewer trials with a quieter objective over
   many trials with a loud one — the search is only as good as its ability to rank.
3. **Refuse to distinguish inside the noise floor.** Report the top-k configurations and
   their spread rather than a single "best". If trials 1–10 are within the CI of each other,
   the honest statement is that the search identified a *region*, not a point.

## Proposed design

### Storage — must survive instance death

Non-negotiable, and learned today: seed-replication runs were lost when the vast.ai instance
was shut down mid-run. Optuna must use a **persistent RDB storage** on a path that gets
synced, not in-memory:

```python
storage = "sqlite:////workspace/optuna/wor_study.db"
study = optuna.create_study(
    study_name=f"wor_{arch}", storage=storage,
    load_if_exists=True,             # resume after a killed instance
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    sampler=optuna.samplers.TPESampler(seed=0),
)
```

`load_if_exists=True` plus a synced `.db` means a shut-down instance costs the in-flight
trial only. Add `optuna/` to the `sync_experiments.py` include groups.

### Pruning — the trainer already emits what is needed

`wor_trainer` logs `val_loss` per epoch, so `trial.report(val_loss, epoch)` +
`trial.should_prune()` slots in with no restructuring. `MedianPruner(n_warmup_steps=5)`
because the LR warmup (13.6b) means the first few epochs are not yet informative — pruning
before warmup finishes would kill exactly the configurations warmup was added to rescue.

### Search space

Per-architecture, and deliberately including the three settings from 13.6:

| Parameter | Range | Why |
|---|---|---|
| `lr` | 1e-4 … 3e-3 log | 3e-4 inherited from the conv head, never revisited |
| `grad_clip` | 1.0 … 50.0 log | 93% of qwen batches clipped at the current 5.0 |
| `warmup_frac` | 0.0 … 0.20 | 13.6(b); 0.0 tests whether warmup is needed at all |
| `weight_decay` | 1e-6 … 1e-2 log | interacts with the no-decay group from 13.6(c) |
| `batch_size` | {16, 32, 64, 128} | interacts with lr; keep lr/batch jointly sampled |
| `lateral_loss_weight` | 1.0 … 6.0 | currently 3.0; the PID steers off lateral only (13.16) |
| `vision_grid` | {4, 8} | the confirmed mechanism; grid 8 still unrun |
| `dropout` / stochastic depth | 0.0 … 0.2 | currently 0.0 everywhere |

Keep `model_size` **out** of the search and sweep it separately — mixing a discrete
capacity axis into a continuous search wastes trials and makes the size question
uninterpretable.

### Fairness protocol (the part that makes it a thesis result)

- Same number of trials per architecture, same search space where applicable, same seed for
  the sampler, same fold structure.
- Report both: **best-tuned vs best-tuned**, and **each architecture under the other's
  best hyperparameters**. The second is the direct quantitative answer to 13.6 — it measures
  how much of any margin is architecture and how much is hyperparameter fit.

### Implementation sketch

New file `tune_wor.py` (keep under the 500-line limit):

```python
def objective(trial, arch, args):
    hp = dict(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        grad_clip=trial.suggest_float("grad_clip", 1.0, 50.0, log=True),
        warmup_frac=trial.suggest_float("warmup_frac", 0.0, 0.2),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        lateral_loss_weight=trial.suggest_float("lateral_loss_weight", 1.0, 6.0),
        vision_grid=trial.suggest_categorical("vision_grid", [4, 8]),
    )
    scores = []
    for fold in range(args.num_folds):          # quieter objective, see defence 2
        trainer = WorldOnRailsTrainer(..., fold=fold, num_folds=args.num_folds, **hp)
        for epoch in range(args.epochs):
            trainer.train_epoch(epoch)
            v = trainer.validate()["val_loss"]
            trial.report(v, epoch + fold * args.epochs)
            if trial.should_prune():
                raise optuna.TrialPruned()
        scores.append(v)
    return float(np.mean(scores))
```

Reuses `WorldOnRailsTrainer` directly rather than shelling out to `train_wor.py`, so pruning
can act mid-run. MLflow stays as-is — log the trial number as a tag so a study and its runs
can be cross-referenced.

### Checklist

- [ ] Add `optuna` to `requirements.txt` (not currently present).
- [ ] Add a **three-way route split** (train/val/test) — currently only train/val exists.
      This is a prerequisite, not an optional extra.
- [ ] Write `tune_wor.py` with the objective above.
- [ ] Persistent SQLite storage under `/workspace/optuna/`, added to `sync_experiments.py`.
- [ ] Verify resume-after-kill actually works (kill a study mid-trial, restart, confirm it
      continues) — this is the failure mode that already cost runs once.
- [ ] Run equal-budget studies for `cnn` and `qwen30m`.
- [ ] Re-score the two winners on the untouched test split with the paired bootstrap from
      `check_val_noise.py`.
- [ ] Cross-apply hyperparameters (each arch under the other's best) for the 13.6 answer.
- [ ] Document as challenges section 13.23 / a new group.

---

# Part 2 — External resources assessed

Verdicts against what this project actually needs: **route-plan-conditioned waypoint
regression, with many independent routes** (routes, not frames, are the statistical unit —
13.19).

## Useful

### `autonomousvision/PDM_Lite_Carla_LB2` — already in use, and not exhausted
The highest-value data available is the **rest of what we already have**. 4 of 8 towns are
downloaded (656 routes). Remaining: Town04 (20.3 GB), Town05 (26.6 GB), then Town12/Town13
(~117 GB each). Already in the exact expected format with route plans — zero loader work.
- [ ] `python download_pdm_lite.py --towns Town04,Town05 --reserve-gb 30` → ~1,840 routes
- [ ] Town12/Town13 only on a large-disk instance; each is a big map with many routes.

### `jkdxbns/autonomous-driving-carla` — CARLA-domain pretrained backbone
YOLO11n (detection: vehicles, pedestrians, traffic lights, speed signs) + **UFLD lane
detection with a ResNet-18 backbone trained on CARLA**, CARLA 0.9.15, MIT, weights on the
Hub via `hf_hub_download()`.

The interesting part is the **UFLD ResNet-18**: our encoder is ImageNet-pretrained
ResNet-34, and Challenge Group 8 is entirely about sourcing better pretrained perception. A
backbone trained on CARLA lane geometry is plausibly a much better frozen feature source for
a lane-following waypoint task than ImageNet classification features — and swapping it stays
inside the "use pretrained perception, don't train vision" scope.
- [ ] Load UFLD weights through the existing `--weights_path` path (same mechanism as the
      PCLA/LAV checkpoints) and A/B against ImageNet ResNet-34.
- [ ] Caveat: ResNet-18 at 800×288 input vs our 256×256/ResNet-34 — feature map geometry
      changes, so `vision_grid` and `PretrainedVisionEncoder.out_channels` need checking.
- Detection/lane *outputs* as auxiliary policy inputs would be a larger architectural change
  and a separate question; not recommended before the items in Tier 1.

### `immanuelpeter/carla-autopilot-multimodal-dataset` — as a cross-source test set only
365 GB, 82,600 frames, MIT, ego state + controls + RGB/segmentation/LiDAR.
**Only 30 autopilot runs**, i.e. ~30 independent routes — far fewer than PDM-Lite's 656, so
it is *worse* than what we have for the statistical-power problem, and it has no route plan
(our policy's only working navigation input, since PDM-Lite's command enum is constant —
9.5). Not a training-data candidate.

Its one real value: it was collected by a **different pipeline**, so evaluating a
PDM-Lite-trained model on it is a genuine out-of-distribution generalisation test. "The
transformer's advantage transfers to data it was not collected alongside" is a strong thesis
claim that no amount of extra PDM-Lite towns can support.
- [ ] Stretch goal, after Tier 1. Requires deriving a route proxy, which risks leakage —
      design carefully or restrict to a control-prediction metric.

## Not useful for this project

| Resource | Verdict |
|---|---|
| `DriveFusion/DriveFusion-Data` | 220 GB VLA dataset — language annotations, VQA, commentary, for vision-language-action models. Different problem. Relevant only if the thesis pivots to VLA. |
| `mkxdxd/carla-dataset` | 2.59 TB, 6,049 scenes, RGB + depth only. **No actions, no waypoints, no commands** — a perception dataset. Its only use would be vision pretraining, which is out of scope. |
| `tianlezeng/CarlaAIr-v0.1.7` | CARLA + AirSim in one process for air-ground (drone + vehicle) research. Not related to this thesis. |
| OpenEnv / TRL `carla_env` (HF blog) | Ports `carla-env` to OpenEnv for GRPO training of *language* models via a tool interface (`observe()`, `emergency_stop()`, `lane_change()`). Not a continuous-control gym env; this project already has `src/envs/vector_carla_env` with PPO. Worth a related-work citation, not an integration. |

## Summary

Of nine links, one is already the backbone of the project and under-exploited (PDM-Lite —
half the towns unused), one offers a genuinely promising swap (CARLA-trained UFLD backbone),
one is a possible OOD test set with caveats, and the rest solve different problems.

**The cheapest high-value action remains downloading the remaining PDM-Lite towns** — same
lever that turned an unresolvable 3% into a confirmed 13.6% margin in six minutes.
