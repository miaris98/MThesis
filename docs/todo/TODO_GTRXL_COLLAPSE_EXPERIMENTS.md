# GTrXL+EZ2 Input-Invariance: Experiment Tracker

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
| E12 (E10 + GTrXL gating on) | done | probe_rel_std=0.00034 (10x worse) | **gating destroys it even harder** |
| E13 (control, legacy init) | done | probe_rel_std=0.00896 | matches S-037 baseline, confirms A/B valid |

**Root cause, narrowed**: no single lever fixes it. The combination of (1) fixed actor-head init,
(2) critic detached from the shared trunk, (3) entropy bonus removed is what moves the needle
(3.4x over control) -- but even that is still ~8x below the healthy random-init reference
(0.243). Both the EfficientZero v2 auxiliary losses and GTrXL's GRU gating actively fight
against whatever gain the fix produces, even at bg_init=0.0. See struggle-solutions S-038.

**Baseline to beat** (from S-037, simplified config @ 15k steps):
per-action logit std-dev `[0.00049, 0.00015, 0.00026, 0.00038]`, argmax constant across all inputs.
**Best so far (E10)**: probe_logit_rel_std=0.03008, still ~8x below healthy (~0.24).

---

## Round 2 -- pushing E10's recipe further

### [ ] E14. E10 recipe, LR=5e-4 (moderate, between E10's 1e-3 and default 2.5e-4)
Tests whether E9's "high LR alone is destructive" finding was really about LR magnitude, or
about LR combined with entropy/critic-coupling still active.

### [ ] E15. E10 recipe, LR=2e-3 (push higher with the other fixes in place)
Maps the LR-sensitivity curve now that entropy/critic confounds are controlled.

### [ ] E16. E10 recipe + EZ2 reward-loss ONLY (isolate from consistency/value-unroll)
Tests whether reward prediction alone (not the SimSiam consistency loss) still hurts.

### [ ] E17. E10 recipe + EZ2 consistency-loss ONLY
Directly isolates the SimSiam-representation-collapse hypothesis (S-036's `SimLoss` hit -0.9999
within ~2,500 steps in every run that enabled it).

### [ ] E18. E10 recipe + consistency-loss at reduced weight (0.1 instead of 0.5)
Tests whether a smaller weight keeps some EZ2 benefit without the collapse.

### [ ] E19. E10 recipe, but vf_coef=0.1 instead of full detach_critic
Cheaper partial-decoupling alternative -- tests if softening the critic's pull is enough.

### [ ] E20. E10 recipe extended to 40k steps
Tests whether probe_rel_std keeps climbing over a longer horizon now that the throttling
factors are removed, rather than plateauing like every pre-E10 config did.

### [ ] E21. E10 recipe, advantage normalization disabled
Near-zero raw advantages divided by a tiny std could itself inject high-variance noise into
the policy gradient. Tests removing the `(adv - mean) / std` normalization step entirely.
