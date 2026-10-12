# Off-policy MCTS agent: time/space complexity and sample-efficiency levers

**Goal (user, 2026-09-23)**: Breakout score >= 30 (human 30.5) within the Atari-100k regime, as early as possible (10k/20k-step screens). Current best: 7.90 @10k, 11.00 @20k (seed 1, 32 sims, S053c). Reference: EfficientZero V2 = 400.1 @100k (paper).

Notation: E = parallel envs (8), S = simulations (32), A = actions (4), B = batch (128), K = unroll steps (5), RR = replay ratio (updates per env step), T = visual tokens (121), d = embed dim (256), L = transformer depth (4).

## 1. Where the time goes (per env step)

| Stage | Work per env step | Complexity | Notes (code) |
|---|---|---|---|
| Collection search | 1/E of a `search_batch` over E roots | O(S * (E * depth * A) Python + 3 GPU syncs per sim) / E | `mcts_engine.py:255-307`: every simulation walks E trees in Python (`MCTSNode` dicts, `_select_child` loops over A children), then one tiny batched dynamics call plus **3 host-device syncs** (`.cpu()` on priors, rewards, values). At E=8 the GPU call is launch-bound; the cost is ~S * (Python + sync latency), not FLOPs. |
| Replay sampling | RR updates, each assembling B windows | O(B * K) Python iterations, **O(B*(K+1)*4*84*84) float32 = ~86 MB** built on CPU and copied to GPU per update | `sample_trajectories`: per-sample Python loops, `astype(np.float32)/255` on the CPU, then a 4x larger host->device copy than the uint8 data needs. |
| Update forward/backward | RR * [encoder on B roots (grad) + encoder on B*K target frames (no grad) + K dynamics steps] | encoder: O(L * T^2 * d) per image for attention, **x(1+K) = 6 images per sample** | T=121 tokens (IMPALA 11x11 grid). Target encoding of K future frames dominates GPU time; EfficientZero encodes the same targets but with a ResNet to a 6x6 latent (T=36 equivalent), ~11x cheaper attention. |
| Reanalyze (optional) | RR * r*B extra searches | O(RR * r * B * S) sims per env step | Same Python search, now over up to 128 roots per update: the reason r>0.25 ran at ~1 step/s (S-055). |
| Evaluation | every 5k steps, N episodes | O(N * episode_len * S) **sequential single-root searches** | `evaluate_agent_mcts` uses `search_single` one env at a time; a 20-episode eval at 25k took >20 min (S-054). |

## 2. Space

| Item | Size | Issue |
|---|---|---|
| Replay obs | capacity * 4 * 84 * 84 B = 28 KB/transition, 50k -> 1.41 GB | Stores full 4-frame stacks: **4x redundant** (consecutive stacks share 3 frames). |
| Replay capacity | `buffer_capacity=50_000` | **Below the 100k budget**: from 50k steps on, the oldest half of the data is overwritten. EfficientZero keeps every transition (1M buffer). Irrelevant at 10k/20k, a real sample-efficiency loss for Gates 3-4. |
| Per-update host batch | ~86 MB float32 | 4x larger than uint8; pure overhead. |

## 3. Sample efficiency: what is missing vs EfficientZero / data-efficient RL

Ranked by expected gain per unit of effort.

1. **No data augmentation.** Every sample-efficient Atari-100k method (DrQ, SPR, EfficientZero, EZ-V2 `augmentation: ['shift','intensity']`) applies random-shift (pad 4, random crop) and intensity augmentation to observations used by the representation/consistency losses. With ~10k transitions a 4-layer transformer over 121 tokens can memorise the buffer; augmentation is the standard fix and costs almost nothing on GPU. **Not implemented in our trainer.**
2. **Low update budget per env sample.** RR 0.5 x B 128 = 64 sampled windows per env step; EZ-V2 does ~1 update of B 256 per transition = 256, 4x more. Raising RR is only affordable once the per-update cost (rows 2-3 above) comes down.
3. **Policy targets from plain visit counts.** With S=32 and forced first visits, visit counts are a coarse policy-improvement operator; EZ-V2 uses Gumbel search with completed-Q targets, which gives a valid improvement signal with 16 sims.
4. **Stale policy targets** (reanalyze): implemented (S-055) but its per-update cost limits it to <=25%.
5. **Scalar value/reward regression** instead of categorical supports.
6. **Replay capacity < budget** (Section 2).

## 4. Plan

Round 2 of the headroom experiments (S-057) adds, behind flags, in this order:
- **Fast sampler**: vectorised window selection and gathering, uint8 batches, normalisation on the GPU (removes the per-update Python loops and 3/4 of the host->device traffic). Pure speed; identical samples.
- **Augmentation** `--augment shift|shift_intensity` on the root observation and on the consistency-target frames, independently per image (as in SPR/EfficientZero).
- **Timing instrumentation**: seconds per 1k steps split into collect / sample / update / eval, so later choices use measured numbers, not the estimates above.
- `buffer_capacity >= total_steps` automatically.
Later rounds: batched evaluation (all eval episodes as one `search_batch`), Gumbel/completed-Q policy targets, categorical supports, 6x6 token downsampling.
