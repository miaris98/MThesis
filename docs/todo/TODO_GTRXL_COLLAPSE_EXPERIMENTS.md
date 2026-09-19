# GTrXL+EZ2 Input-Invariance: Experiment Tracker

**INVESTIGATION CLOSED (S-043, 2026-09-19)**: ROOT CAUSE CONFIRMED AND FIXED on the real
architecture. `ImpalaCNNEncoder` had no explicit weight init by default; adding
`NatureCNNEncoder`'s own `kaiming_normal_`(fan_out/relu)+`trunc_normal_` scheme
(`cnn_kaiming_init=True`) alone -- with NO other E5-E51 compensating fixes, at original
hyperparameters -- broke `ImpalaGTrXLAgent` out of the input-invariance collapse:
`S043a_kaiming_alone_300k` scored `[0,0,0,0,0,18,8,4,7,9]` (best 18.0),
`probe_logit_rel_std=0.979` (vs E10's flat 0.030, matching the I0/I1b healthy ~1.0 reference).
Confirms the I1b finding (on the simplified `QwenAtariActorCritic` testbed) generalizes to the
real GTrXL+EZ2 stack. **Important twist**: stacking E10's compensating fixes (detach_critic,
no entropy, 1e-3 LR) ON TOP of the real fix (`S043b`) actively re-collapses the policy
(`probe_logit_rel_std=0.00018`, effectively flat) -- those fixes were compensation for the
broken encoder, not a generally good recipe, and become harmful once the actual bug is fixed.
**Conclusion**: the correct go-forward recipe is `cnn_kaiming_init=True` alone at original
hyperparameters, not the E10 stack. Most of E5-E51's hyperparameter tuning below was chasing
downstream symptoms of this one missing init call. See struggle-solutions S-043 for full
results tables and the closing interpretation. **Open follow-up**: S043a's trajectory is
noisier than I0/I1b's clean climb (late start, peak-then-partial-decay) -- a longer/cleaner
confirmation run (ideally with GRU gating and EZ2 aux losses re-enabled) is the recommended
next step before moving on to Atari 100k Benchmark harness work.

**IMPORTANT METHODOLOGICAL CAVEAT (S-042, 2026-09-19)**: the incremental-integration test found
that even the PROVEN `QwenAtariActorCritic` + `train_ppo.py` baseline (S-029's genuine 0.00->17.00
climb) shows `probe_logit_rel_std=0.013` and a flat 0.0 eval score at the 15k-step budget used
throughout this entire tracker -- below E10's 0.03008 and nowhere near the 0.243 healthy-reference
target. **Every relative ranking below ("E10 beats X", "E37 beats E10") is still valid as a
same-budget comparison, but the absolute "still far from healthy" framing needs revision** -- the
proven architecture doesn't clear that bar at 15k steps either. See struggle-solutions S-042
before drawing conclusions from probe_logit_rel_std numbers in isolation.

**Problem statement.** The `ImpalaGTrXLAgent` actor produces essentially identical logits for
wildly different observations (per-action logit std-dev ~`0.0002-0.0005` across blank / white /
noise / real gameplay frames). Every eval score in this investigation (the recurring `0.00` and
`11.00`) is an artifact of a constant-action policy being steered by the eval harness's scripted
`FIRE`-on-life-loss override against a deterministic emulator -- not learning. See
`docs/design/struggle-solutions.md` S-024 through S-037 for the full trail.

**Already ruled out** (all still fully input-invariant at 15k-60k steps):
| # | Intervention | Entry | Result |
|---|---|---|---|
| - | Off-policy replay buffer + PPO-clip + GAE | S-030/S-031/S-032 | invariant |
| - | On-policy PPO+GAE rewrite | S-033/S-034 | invariant |
| - | Per-env independent eps-greedy | S-033 | invariant |
| - | `bg_init` 2.0 -> 0.0 (open the GRU gates) | S-035/S-036 | invariant |
| - | Remove GRU gating entirely + disable all EZ2 aux losses | S-037 | invariant |

---

## Phase A -- Diagnostics (cheap, no training)

### [ ] E1. Layer-wise variance probe (trained checkpoint)
**Hypothesis**: the signal dies at a specific, identifiable layer, not diffusely.
**Method**: feed N very different observations through a trained checkpoint and measure, at every
stage (raw obs -> CNN stage1/2/3 -> proj -> encoder LayerNorm -> each GTrXL block -> latent_z /
policy_repr -> logits), the across-input std-dev, both absolute and relative to activation scale.
**Success criterion**: pinpoint the first stage where relative across-input variation collapses.

### [ ] E2. Same probe on a randomly-initialized (untrained) network
**Hypothesis**: distinguishes "born invariant" (architecture/init bug) from "trained into
invariance" (optimization pathology).
**Success criterion**: if random-init shows healthy variance and trained does not, training is
actively destroying the signal -- which points at the loss, not the architecture.

### [ ] E3. Reference measurement on a known-healthy checkpoint
**Hypothesis**: we have no idea what a *healthy* logit std-dev even looks like for this task.
**Method**: run the same probe against S-029's from-scratch `QwenAtariActorCritic` PPO run (which
genuinely climbed 0.00 -> 17.00) if its checkpoint is still on the box; else the historical BC+PPO
checkpoint at `E:\MThesis_EXP\atari_qwen\qwen_bc_ppo_finetuned\`.
**Success criterion**: a concrete target number to compare every other experiment against.

### [ ] E4. Supervised sanity check -- can this architecture learn ANYTHING input-dependent?
**Hypothesis**: the strongest possible test of a structural forward/backward bug.
**Method**: take the exact `ImpalaGTrXLAgent`, and train the actor head with plain cross-entropy
to map all-black frames -> action 0 and all-white frames -> action 1, for a few hundred gradient
steps. No RL, no sparse reward, no replay -- a task a working network must solve almost instantly.
**Success criterion**: if it cannot separate two maximally-different inputs, there is a structural
bug and no amount of RL tuning or step budget will ever help. If it can, the architecture is sound
and the problem is the RL signal/optimization balance.

---

## Phase B -- Targeted fixes (15-20k step runs, measured by probe + score)

### [ ] E5. Fix the actor-head initialization (**prime suspect**)
**Bug**: `ImpalaGTrXLAgent._init_weights` applies `orthogonal_(gain=0.01)` to **every** Linear in
`actor_head`, which is `Sequential(Linear(256,256), GELU, Linear(256,action_dim))` -- so both
layers, compounding to roughly `1e-4` attenuation. Standard practice (CleanRL) applies the small
`0.01` gain only to the **final** layer, with `sqrt(2)` on hidden layers. Consequences: (a) logits
are crushed toward a constant, (b) gradient flowing back into the shared trunk is scaled by ~`5e-5`,
so the actor's learning signal is negligible compared to the critic's.
**Fix**: `sqrt(2)` on the hidden layer, `0.01` on the output layer only.

### [ ] E6. Stop the critic from flattening the shared trunk
**Hypothesis**: `critic_head` is initialized at `gain=1.0` on both layers -- ~1e4x stronger
gradient into the shared trunk than the actor's. With Breakout's sparse rewards the value target is
near-constant early on, so the critic actively pushes the trunk toward a constant representation,
and the attenuated actor cannot counteract it.
**Method**: detach the critic from the trunk (or give it its own trunk / scale down `vf_coef`) and
re-measure input variance.

### [ ] E7. Learning-rate sweep
**Hypothesis**: the network is moving, just far too slowly to show up in 15k steps.
**Method**: 10x LR (2.5e-3) and 3x LR at 15k steps; compare logit std-dev trajectory.

### [ ] E8. Encoder LayerNorm / feature-scale audit
**Hypothesis**: `ImpalaCNNEncoder` has **no weight initialization at all** (unlike
`NatureCNNEncoder`, which has `_init_weights`), and ends in a `LayerNorm` over `embed_dim` after
projecting from only 32 channels. Either could be washing out spatial/input-dependent signal.
**Method**: add proper CNN init; A/B the final LayerNorm.

### [ ] E9. Isolate the SimSiam consistency-loss collapse
**Observation**: `SimLoss` hits `-0.9999` (perfect cosine similarity) within ~2,500 steps in every
run that enables it -- the textbook signature of SimSiam **representation collapse**, where the
easiest way to predict the next latent is to make all latents identical. That is exactly an
input-invariance generator.
**Method**: run with consistency weight on vs. off and probe latent variance directly (S-037
disabled it wholesale but never measured latent variance specifically).

### [ ] E10. Fix the evaluation harness
**Bug**: `evaluate_agent` forces `action = 1` (FIRE) on every life loss and after 30 rewardless
steps, so a constant-action network produces a plausible-looking, perfectly reproducible score.
Combined with a fixed seed, no sticky actions, and deterministic argmax, every eval is one frozen
trajectory. This is what made `11.00` look like learning for days.
**Fix**: remove the scripted override (or make it opt-in), sample stochastically from the policy,
use several seeds, and report the spread.

---

## Phase C -- Scale-up (only after Phase B identifies a fix)

### [ ] E11. Long-horizon run (200k-300k steps)
Matches S-029's own successful budget. Only worth spending once a Phase B experiment shows the
probe's variance actually rising.

### [ ] E12. Reward/advantage signal audit at scale
Log advantage magnitudes, entropy, and clipfrac over a long run to confirm the policy gradient
carries real signal rather than noise around zero.

---

## Results log

| Exp | Status | Key number | Verdict |
|---|---|---|---|
| E1 (layer probe, trained) | done | rel_std collapses 0.307->0.072->0.0085 through trunk+head | signal dies gradually, worst at actor_head |
| E2 (layer probe, random init) | done | policy_repr rel_std=0.307 (healthy) | network is NOT born broken |
| E3 (reference measurement) | skipped | - | superseded by E2's random-init reference |
| E4 (supervised sanity fit) | done | solved in ~45 steps, logit gap 9.68 | **no structural bug** -- architecture can learn |
| E5 (fix actor-head init alone) | done | probe_rel_std=0.00704 | not sufficient alone |
| E6 (E5 + detach critic) | done | probe_rel_std=0.01456 (1.6x) | helps, not enough alone |
| E7 (E5 + no entropy) | done | probe_rel_std=0.00891 (~1x) | not sufficient alone |
| E8 (E5 + anneal entropy) | done | probe_rel_std=0.01812 (2.0x) | helps, not enough alone |
| E9 (E5 + 10x LR alone) | done | probe_rel_std=0.00001 (**worse**) | high LR alone is destructive |
| **E10 (detach+no-ent+lr1e-3)** | **done** | **probe_rel_std=0.03008 (3.4x)** | **best combo so far** |
| E11 (E10 + EZ2 aux losses on) | done | probe_rel_std=0.00739 (reverts) | **EZ2 losses destroy the gain** |
| E11b (long-horizon 100k steps) | done | LogitSpread pinned at 0.00000–0.00029, degenerate actions (100% FIRE / 97% LEFT) | **Step budget alone does NOT break collapse.** Confirms that more gradient updates under current RL/aux objective cannot escape constant-action equilibrium. |
| E12 (E10 + GTrXL gating on) | done | probe_rel_std=0.00034 (10x worse) | **gating destroys it even harder** |
| E13 (control, legacy init) | done | probe_rel_std=0.00896 | matches S-037 baseline, confirms A/B valid |

**Root cause, narrowed**: no single lever fixes it. The combination of (1) fixed actor-head init,
(2) critic detached from the shared trunk, (3) entropy bonus removed is what moves the needle
(3.4x over control) -- but even that is still ~8x below the healthy random-init reference
(0.243). Both the EfficientZero v2 auxiliary losses and GTrXL's GRU gating actively fight
against whatever gain the fix produces, even at bg_init=0.0. Furthermore, E11b proves conclusively
that scaling the step budget to 100k steps does not break the collapse. See struggle-solutions S-038.

**Baseline to beat** (from S-037, simplified config @ 15k steps):
per-action logit std-dev `[0.00049, 0.00015, 0.00026, 0.00038]`, argmax constant across all inputs.
**Best so far (E10)**: probe_logit_rel_std=0.03008, still ~8x below healthy (~0.24).

---

## Round 2 -- pushing E10's recipe further

### [x] E14. E10 recipe, LR=5e-4 (moderate, between E10's 1e-3 and default 2.5e-4)
Tests whether E9's "high LR alone is destructive" finding was really about LR magnitude, or
about LR combined with entropy/critic-coupling still active.
**Result**: probe_logit_rel_std=0.01593 -- worse than E10. Lower LR alone doesn't help either.

### [x] E15. E10 recipe, LR=2e-3 (push higher with the other fixes in place)
Maps the LR-sensitivity curve now that entropy/critic confounds are controlled.
**Result**: probe_logit_rel_std=0.00032 -- much worse. High LR is destructive even with
entropy/critic fixed in place; E9's finding generalizes.

### [x] E16. E10 recipe + EZ2 reward-loss ONLY (isolate from consistency/value-unroll)
Tests whether reward prediction alone (not the SimSiam consistency loss) still hurts.
**Result**: probe_logit_rel_std=0.00074 -- much worse.

### [x] E17. E10 recipe + EZ2 consistency-loss ONLY
Directly isolates the SimSiam-representation-collapse hypothesis (S-036's `SimLoss` hit -0.9999
within ~2,500 steps in every run that enabled it).
**Result**: probe_logit_rel_std=0.00152 -- much worse.

### [x] E18. E10 recipe + consistency-loss at reduced weight (0.1 instead of 0.5)
Tests whether a smaller weight keeps some EZ2 benefit without the collapse.
**Result**: probe_logit_rel_std=0.02672 -- close but still below E10.

### [x] E19. E10 recipe, but vf_coef=0.1 instead of full detach_critic
Cheaper partial-decoupling alternative -- tests if softening the critic's pull is enough.
**Result**: probe_logit_rel_std=0.00098 -- much worse. Full detach is necessary.

### [x] E20. E10 recipe extended to 40k steps
Tests whether probe_rel_std keeps climbing over a longer horizon now that the throttling
factors are removed, rather than plateauing like every pre-E10 config did.
**Result**: probe_logit_rel_std=0.00186 -- worse, does not climb. Re-confirms E11b's
long-horizon finding at a smaller scale.

### [x] E21. E10 recipe, advantage normalization disabled
Near-zero raw advantages divided by a tiny std could itself inject high-variance noise into
the policy gradient. Tests removing the `(adv - mean) / std` normalization step entirely.
**Result**: probe_logit_rel_std=0.00629 -- worse.

---

## Round 3 -- 30 Architectural, Representation & Signal Levers (E22–E51)

### Category A: Attention & Spatial Aggregation (Query Bottleneck)
- [ ] **E22. Cross-Attention Query Bottleneck (Perceiver/DETR style)**: Instead of concatenating `[STATE, POLICY]` with 121 visual tokens in full self-attention (where 121 tokens dilute the query), use explicit cross-attention where `POLICY` queries the visual memory tokens as keys/values.
- [ ] **E23. Attention Temperature Sharpening**: Attention weights over 121 tokens are near-uniform ($1/121 \approx 0.008$). Scale $Q \cdot K^T / (\sqrt{d} \cdot \tau)$ with temperature $\tau < 1.0$ (e.g. 0.5) to sharpen spatial focus on the ball and paddle.
- [ ] **E24. Global Spatial Pooling (GAP) Bypass**: Compare the 121-token sequence against global average pooling or flattened projection `Linear(32*11*11, embed_dim)` (NatureCNN style), testing whether the 121-token attention mechanism is diluting spatial localization.
- [ ] **E25. Residual CNN-to-Actor Bypass**: Add a direct residual connection from the visual CNN encoder directly to `actor_head` (`policy_repr = transformer_out + cnn_proj(vis_tokens.mean(1))`), preventing the transformer trunk from acting as an information bottleneck.
- [x] **E26. Remove Final LayerNorm on `policy_repr`**: `self.norm(seq)` immediately before `self.actor_head` normalizes across the feature dimension, potentially crushing the variance of low-magnitude spatial activations.
  **Result**: `norm_policy_repr=False` on E10 -> probe_logit_rel_std=0.00335 -- worse than E10.

### Category B: Representation Collapse & Auxiliary Loss Redesign
- [ ] **E27. VICReg Variance Regularizer on `latent_z` & `policy_repr`**: Add explicit variance penalty $L_{var} = \max(0, 1 - \sqrt{\mathrm{Var}(z) + \epsilon})$ from VICReg (Bardes et al.) to mathematically forbid representation collapse.
- [ ] **E28. Stop-Gradient Verification in Target Latent Projection**: Verify and enforce `stop_gradient` on the target projection in the SimSiam branch. Without stop-gradient, SimSiam trivially minimizes cosine distance by collapsing representations to a constant.
- [ ] **E29. Decouple Predictor Trunk from Actor Trunk**: Give EfficientZero v2's multi-step dynamics predictor its own projection head rather than backpropagating auxiliary consistency gradients directly into the actor's representation.
- [ ] **E30. Replace Cosine Similarity with InfoNCE / Contrastive Loss**: Cosine loss has no negative samples and readily collapses. InfoNCE with temporal negatives (frames from other timesteps or other envs in the batch) forces distinct representations.
- [x] **E31. Dynamic Aux Loss Warmup**: Set `aux_loss_weight = 0.0` for the first 20k steps, allowing the policy and value functions to establish basic ground-truth dynamics before turning on auxiliary predictive losses.
  **Result**: `aux_warmup_steps=10000` on E10+EZ2(on) at 15k steps -> probe_logit_rel_std=0.02334 -- close but below E10, and clearly better than E11's 0.00739 (EZ2 at full strength from step 1), so warmup does help relative to no-warmup EZ2, just not enough to beat E10 (EZ2 off) outright at this step budget.

### Category C: Exploration & Reward Signal in Sparse Atari Breakout
- [ ] **E32. Random Network Distillation (RND) Intrinsic Curiosity**: Add an RND exploration bonus $r_{int} = \|\hat{f}(s) - f(s)\|^2$. In Breakout, novel states (ball bouncing, bricks breaking) yield high intrinsic reward, breaking the $r=0$ dead zone.
- [x] **E33. Living / Ball-in-Play Reward Shaping**: Add a tiny living reward ($+0.005$ per alive frame) or ball-in-play reward to provide dense gradient signal before bricks are struck.
  **Result**: `living_reward=0.005` on E10 -> probe_logit_rel_std=0.00273 -- worse than E10.
- [ ] **E34. Ball-Paddle Alignment Heuristic Reward**: Use frame differencing or a heuristic proxy reward for paddle horizontal alignment with the ball during early training to bootstrap intercept trajectories.
- [x] **E35. Softmax Action Sampling in Evaluation (Remove Argmax Lock)**: In `evaluate_agent()`, evaluate with stochastic sampling ($\tau = 0.5$ or $\tau = 1.0$) rather than deterministic `argmax`. Under `argmax`, a 0.0002 logit advantage locks into 100% constant action.
  **Result**: `eval_temperature=1.0` on the E10 recipe -> score reads **1.2 on all 3 eval checkpoints** instead of a hard 0.0/11.0 toggle. First evidence the deterministic-argmax eval harness was itself partly manufacturing the "11.00 looks like learning" artifact -- though 1.2 is still near-random for Breakout, not real learning. See struggle-solutions S-039.
- [x] **E36. Sticky Action & Frame-Skip Evaluation**: Evaluate with standard Atari sticky actions ($p=0.25$) to break deterministic emulator looping where a single action locks into a static cycle.
  **Result**: `sticky_action_p=0.25` on the E10 recipe -> score still locks to a constant 11.0 on all 3 checkpoints. The underlying policy is degenerate enough that sticky actions alone don't expose it (unlike E35's temperature sampling).

### Category D: Optimization, Advantage & Gradient Dynamics
- [x] **E37. Asymmetric Learning Rates (Actor LR = 5x Critic LR)**: Invert the standard ratio: set `actor_lr = 1e-3` and `critic_lr = 2e-4` to prevent the critic from dominating the trunk.
  **Result -- BREAKTHROUGH, the best result in the entire investigation.** Original E37 (actor_lr=1e-3, rest=2e-4, 5x ratio, seed=42, 15k steps): probe_logit_rel_std=0.03133 (barely past E10) but probe_**policy_repr**_rel_std=0.1218 -- ~4x every other result including E10. Follow-up batch confirmed and blew past this:
  - **E37b** (same 5x-ratio recipe, seed=7): probe_logit_rel_std=**0.15585**, policy_repr_rel_std=**0.19534**.
  - **E37c** (same recipe, 40k steps): probe_logit_rel_std=0.03524, policy_repr_rel_std=0.10085 -- reproduces the original magnitude, more steps helps mildly.
  - **E37d** (ratio pushed to 20x: actor_lr=1e-3, rest=5e-5, seed=42, 15k steps): probe_logit_rel_std=**0.21385**, policy_repr_rel_std=**0.27571** -- within ~10% of the healthy **random-init reference** (logits rel_std=0.243, policy_repr rel_std=0.307), the first result in the whole S-024-S-039 trail to get this close. Eval score also hit 11.0 on one of 3 checkpoints (still not consistently climbing, but no longer a hard 0.0-only lock like most other configs).
  Asymmetric actor/critic LR -- not detach_critic, not entropy removal alone -- looks like the real missing lever; those two were necessary but nowhere near sufficient. See struggle-solutions S-040.
  **Confirmation batch** (E37e: seed=7 at 20x ratio; E37f: 20x ratio at 100k steps): E37e reproduces and even **exceeds** the healthy reference (policy_repr_rel_std=0.31073 vs. 0.307 reference) -- not a lucky seed. But E37f shows the gain **decays at 100k steps** (0.23164, down from the 15k-step ~0.28-0.31 range), and **eval score stayed a flat 0.0 in both** despite the healthy-looking representation. The diagnostic probe and actual RL learning have decoupled: fixing input-invariance (as measured) does not by itself fix score improvement. Likely cause: the 20x-slower trunk/critic LR may be starving the critic's value learning, degrading the GAE advantage signal. **Next**: try an intermediate ratio (between 5x and 20x) and directly monitor critic value-loss/advantage trajectory for signs of critic starvation.

  **Ratio sweep at 10x** (E37g 15k, E37h 40k): policy_repr_rel_std=0.203/0.191 -- a clean midpoint between 5x (0.122) and 20x (0.276-0.311). VLoss trajectories checked directly across 5x/10x/20x: all stay in a similar 0.017-0.064 range with no divergence at higher ratios, so the critic-starvation hypothesis is **not supported**. This is a monotonic dose-response (more decoupling = more variance), not a sweet spot, and eval score stays flat 0.0 at every ratio tested. The bottleneck between "healthy representation" and "climbing eval score" remains unidentified -- see struggle-solutions S-040 for the full writeup and next-step options (very long run vs. PPO-mechanics audit).

  **PPO-mechanics audit** (S-041): diffed against `train_ppo.py`, the trainer that proved `QwenAtariActorCritic` can climb 0.00->17.00. Found and fixed 3 real gaps: no LR annealing (added `lr_schedule` cosine/linear), no value-loss clipping (added `clip_vloss`), and 100x-weaker weight decay (1e-4 vs the proven 1e-2). Tested individually on E10 (all landed below 0.03008 -- Epp1 lr_cosine: 0.01044, Epp2 clip_vloss: 0.00326, Epp3 weight_decay: 0.00881) and all three combined with E37g's 10x-ratio recipe (Epp4: reproduces E37g's own ~0.20 representation variance, still a flat 0.0 eval score). **None of these unlock eval-score improvement.** The PPO-mechanics gap is ruled out as the explanation. Next direction: incremental integration (see the "Module-level regression tests" section below) -- start from the proven `QwenAtariActorCritic` + `train_ppo.py` combination and swap in `ImpalaGTrXLAgent` components one at a time to isolate exactly which swap breaks learning, rather than continuing to vary hyperparameters on a stack that differs from the proven baseline in five places at once.
- [x] **E38. Advantage Normalization Clamp / Soft-Standardization**: Standard advantage normalization $(A - \mu) / (\sigma + 1e-8)$ in sparse reward batches explodes noise when $\sigma \to 0$. Use $(A - \mu) / \max(\sigma, 0.1)$ or unnormalized raw GAE advantages.
  **Result**: `adv_norm_std_floor=0.1` on the E10 recipe -> `probe_logit_rel_std=0.03151`, marginally above E10's 0.03008 -- but eval score was a flat 0.0 on all 3 checkpoints (worse than E10's mixed 0/11), and the margin is inside the run-to-run noise floor seen across nominally-identical configs. **Not confirmed as a real improvement without a repeat run.**
- [x] **E39. GAE Lambda ($\lambda$) Sweep**: Test $\lambda \in \{0.80, 0.90, 0.95, 0.99\}$ (0.95 is E10's default, already covered). Lower $\lambda$ reduces variance in sparse-reward settings where long unrolls accumulate near-zero bootstrap errors.
  **Result**: $\lambda=0.80$ -> probe_logit_rel_std=0.00455 (worse); $\lambda=0.99$ -> 0.01150 (worse). Neither direction helps; 0.95 (E10's default) remains best in this dimension.
- [ ] **E40. Completely Separate Actor and Critic Trunks**: Decouple the actor network and critic network into two distinct IMPALA+Transformer branches, completely eliminating gradient interference between policy and value.
- [x] **E41. Orthogonal Weight Init on Visual Encoder**: Add explicit `orthogonal_(gain=sqrt(2))` initialization to all convolutional layers in `ImpalaCNNEncoder` (which currently use default PyTorch Kaiming uniform).
  **Result**: `cnn_orthogonal_init=True` on E10 -> probe_logit_rel_std=0.00028 -- much worse, actively harmful.

### Category E: Behavioral Cloning & Pretrained Bootstrap
- [ ] **E42. 5k-Step BC Warmup from Expert Dataset**: Pretrain `ImpalaGTrXLAgent` on `data/atari_expert/breakout_expert_100k.npz` with cross-entropy for 5k steps before RL. Guarantees the actor starts with high logit spread and valid paddle movement.
- [ ] **E43. Auxiliary Behavioral Cloning Regularizer during PPO**: Add an auxiliary loss $L_{total} = L_{PPO} + \alpha L_{BC}$ against a small buffer of expert transitions to anchor the policy logits against collapse.
- [ ] **E44. Policy Distillation from Pretrained Qwen PPO Baseline**: Distill from the successful `compare_ppo_qwen` model (which achieved score 17.00) into `ImpalaGTrXLAgent` via KL divergence $D_{KL}(\pi_{qwen} \| \pi_{gtrxl})$.
- [ ] **E45. Off-Policy Q-Learning / SAC-Discrete Head**: Replace the on-policy PPO policy gradient with discrete SAC or Double DQN on top of the GTrXL latent, where TD learning directly optimizes $Q(s, a)$ rather than relying on advantage normalization.
- [ ] **E46. Value-Guided Rollout Exploration**: In training rollouts, sample actions from $\pi(a|s) \propto \exp(Q_{aux}(s, a) / \tau)$ using the 1-step lookahead Q-values from EfficientZero rather than pure $\pi_\theta(a|s)$.

### Category F: Head Architecture & Action Constraints
- [x] **E47. Single-Layer Actor Head (Linear without Hidden Layer)**: Replace `Sequential(Linear(256, 256), GELU, Linear(256, action_dim))` with a direct `Linear(256, action_dim)`. In linear heads, logits are directly proportional to feature projections with no internal dead GELUs.
  **Result**: `single_layer_actor_head=True` on E10 -> probe_logit_rel_std=0.00424 -- worse than E10.
- [ ] **E48. LayerNorm inside Actor Head**: Add `LayerNorm` between the hidden linear and output linear in `actor_head` (`Linear(256, 256) -> LayerNorm -> GELU -> Linear(256, 4)`) to stabilize activation variance.
- [ ] **E49. Spectral Normalization on Critic**: Apply spectral normalization (`torch.nn.utils.spectral_norm`) to the critic head to bound the Lipschitz constant and prevent massive critic gradients from over-regularizing the trunk.
- [ ] **E50. Cosine Classifier Head**: Compute logits as cosine similarity between normalized feature $f$ and normalized action prototypes $w_a$: $\text{logit}_a = \frac{f \cdot w_a}{\|f\| \|w_a\| \cdot \tau}$. Logits are bounded in $[-1/\tau, +1/\tau]$, mathematically preventing logit collapse to a constant scalar.
- [ ] **E51. Discrete Action Masking / Force Fire on Reset Wrapper**: Wrap the environment with `FireResetEnv` and mask out NOOP on Breakout (reducing action space from 4 to 3: FIRE, RIGHT, LEFT), eliminating the degenerate "stay still" stationary point.

---

## Module-level regression tests (safety net, not a collapse experiment)

**Motivation**: the actor-head init bug (E5, S-038) and the input-invariance collapse it
partially caused went undetected for ~15 experiments (S-024-S-037) because every check was an
expensive end-to-end RL run -- a broken module only became visible once its symptom propagated
all the way to a flat eval score. E4's supervised sanity fit (`sanity_supervised_fit.py`) caught
the *first* structural question ("can this network learn anything input-dependent at all?") cheaply
and quickly, but only by testing the whole `ImpalaGTrXLAgent` end-to-end, not each module. A
permanent per-module test suite would catch a class of bugs like E5 (or a future equivalent) in
seconds via `pytest`, before spending GPU-hours discovering it through a collapsed policy.

**Plan**: add `atari_qwen/tests/test_modules.py` (pytest) with one test class per module, run
against every architecture variant that implements it, so a regression in one arch's version of a
module (e.g. `ImpalaCNNEncoder` vs `NatureCNNEncoder`, or `ImpalaGTrXLAgent`'s actor_head vs
`QwenAtariActorCritic`'s) is caught in isolation rather than only showing up as an unexplained
end-to-end score:

- [ ] **Visual encoders** (`ImpalaCNNEncoder`, `NatureCNNEncoder`, `PatchTokenizer` in
  `visual_encoders.py`): feed maximally-different synthetic inputs (all-zero, all-255, random
  noise) and assert output tokens have non-trivial across-input variance (rel_std above some
  floor) -- i.e. run E1/E2's layer probe as an assertion, not a one-off script, for every encoder.
- [ ] **Transformer trunk blocks** (`GTrXLBlock` and any plain-residual variant): assert a
  forward+backward pass changes every parameter's gradient to something non-zero for a
  distinguishing input pair, and that `use_gru_gating=True` vs `False` produce different outputs
  (catches a gate silently defaulting to a no-op, as bg_init effectively did pre-S-035).
- [ ] **Actor/critic heads** (across `ImpalaGTrXLAgent` and `QwenAtariActorCritic`): assert
  gradient magnitude reaching the shared trunk from the actor head is within some sane ratio
  (e.g. 0.1x-10x) of the critic head's -- would have caught the ~65x-100x actor/critic gradient
  asymmetry from S-038 immediately, without needing the E1 layer probe run against a trained
  checkpoint first.
- [ ] **EfficientZero v2 predictor** (`efficientzero_v2_predictor.py`): assert the SimSiam
  consistency loss does NOT trivially reach its collapse minimum (-1.0 cosine similarity) within
  a handful of steps on random targets -- would have flagged the representation-collapse pattern
  noted in S-036 (`SimLoss -> -0.9999` within ~2,500 steps) as a red flag before it ever showed up
  in a real training run.
- [ ] **Cross-architecture parity check**: for any module that exists in more than one
  architecture (visual encoder, actor head), run the same synthetic-input variance assertion
  against each implementation and report them side by side, so a newly-added architecture
  variant is checked against the same bar the others already pass, rather than discovered broken
  only once it's wired into a full training run.

**Non-goal**: these tests check *that a module is structurally capable of producing
input-dependent, gradient-bearing output* -- they are not a substitute for the actual RL
experiments above, which test whether the *training dynamics* (not just the forward/backward
pass) stay healthy. E4's finding (architecture sound, RL signal/balance is the actual problem)
is exactly the distinction: module tests would have still passed throughout S-024-S-037.
