# Deep-research prompt: novel, high-probability ideas for a sample-efficient MCTS agent on Atari 100k

Paste everything below the line into a deep-research LLM (e.g. ChatGPT / Gemini / Claude deep research). It was written on
2026-09-29 from the literature checks in TODO_ACTIVE B19-B28; update the "Already known" section as items get tried.

---

## Role and goal

You are a research scout for a master's thesis in model-based reinforcement learning. Find **research ideas that have not
been published yet** (as far as a thorough search of 2019-2026 literature can tell) **and** that have a **high probability
of producing a clean, reportable result** within the compute budget below. Novelty without a plausible mechanism is not
useful; a well-known trick is not useful either. Rank ideas by (novelty) x (probability of working) / (cost).

## The system (what the ideas must fit)

- **Agent:** our PyTorch port of **EfficientZero V2 (EZ-V2)**, a MuZero-family agent: representation network, latent
  dynamics network, **value-prefix** head (an LSTM over the imagined unroll), value and policy heads, **SimSiam-style
  self-supervised temporal consistency loss**, **Gumbel MuZero** search (sequential halving, 16 simulations), off-policy
  **reanalyze** of targets (about 71% of wall-clock), categorical two-hot value/reward targets through MuZero's invertible
  **h-transform (value rescaling)**, sign-clipped rewards during training.
- **Our variation:** a **gated spatial transformer trunk** ("GTrXL" in our code): after the conv layers of the
  representation and dynamics networks, 2 GTrXL blocks (pre-LayerNorm, **GRU-style gating**, gate closed at init,
  zero-initialised output so the model starts exactly as EZ-V2) mix the 36 spatial tokens (6x6) of one latent state.
  It has **no memory across time and a learned absolute position table**, i.e. only the "gated" half of the
  **Gated Transformer-XL** (Parisotto et al. 2020); the **Transformer-XL** half (segment memory, relative positions,
  Dai et al. 2019) is missing. The comparison arm is EZ-V2's own **ResNet trunk**.
- **Benchmark:** **Atari 100k** (100k agent steps = 400k frames), evaluated without sticky actions per the Atari-100k
  protocol; reported as raw score, human-normalised score (HNS), and ideally **IQM with stratified bootstrap CIs (rliable)**.
- **Current results (one seed, Breakout):** GTrXL 299 / 320 / 309 / 372 at 40k / 50k / 60k / 70k; ResNet 226 / 231 / 326 /
  218 / 378 at 40k-80k. Official EZ-V2 code on our hardware: best 363 at 50k, 321 at 100k (paper: 400 at 100k). At 10k
  steps over 6 seeds: GTrXL 20.2 (SD 11.3) vs ResNet 15.8 (SD 3.5), not separated; GTrXL's seed variance is the main
  statistical problem.

## Constraints

- **Budget:** 1-2 GPUs (24-32 GB), ~14 h per 100k run per seed, two runs per GPU; seeds are the bottleneck, so ideas that
  need 50+ seeds to show an effect are out.
- **No extra environment interaction** beyond 100k steps; no human demonstrations; no large pretrained models trained on
  game data. A frozen, general-purpose pretrained perception model is allowed if it helps an idea.
- The idea must be **implementable in an EZ-V2-style codebase** (change a head, a loss, the trunk, the search, the replay,
  or the targets) in days, not months.
- Prefer ideas where the **transformer trunk matters** (the thesis contrasts attention vs convolution in the world model),
  but a strong trunk-agnostic idea is acceptable.

## Already known or already planned (do not propose these as novel; you may propose a clearly different twist)

Sample efficiency and plasticity: **BBF** (Bigger, Better, Faster), **SR-SPR** (resets, replay ratio), **Hare & Tortoise**
(EMA reset), **PLASTIC** (SAM + reset + LayerNorm + CReLU), **dormant neurons** (Sokar et al. 2023), LayerNorm + weight
decay (Lyle et al. 2024), **SimbaV2**. Targets and losses: **HL-Gauss** ("Stop Regressing", ICML 2024), **TWISTER's
action-conditioned contrastive predictive coding (AC-CPC)**, **GW-PCZero** path consistency, raw rewards vs clipping with
the h-transform (LightZero issue #239, never ablated in the EZ line). Search and speed: **ReZero** (backward-view and
entire-buffer reanalyze), **V-MCTS** (adaptive simulations), **Epistemic MCTS** (ICLR 2025), test-time search scaling.
Architecture and world models: **UniZero** (transformer over latent history with MCTS), **IRIS**, **STORM**, **TWM**,
**DIAMOND**, **SimNorm** (TD-MPC2), **QK-norm / sigma-reparam** against **attention entropy collapse** (Zhai et al. 2023),
**"Don't flatten, tokenize!"** (SoftMoE's gain is tokenisation), **"Mind the GAP"**, **ConViT** (gated positional
self-attention). Symmetry: **SiT** (Symmetry-Invariant Transformers, fixed symmetries, Atari 100k, model-free),
**Equivariant MuZero** (exact known group, ProcGen), **Residual Pathway Priors** (soft equivariance), **Augerino** and
learnable-augmentation symmetry discovery, **partially equivariant RL in symmetry-breaking environments** (2025).
Object-centric: **OC-STORM**, **ObjectZero**, SAM-mask inputs ("Virtual Augmented Reality", no gain on Breakout/Pong).
Interpretability: "What model does MuZero learn?" (2023).
**Our own planned ideas:** a world model that learns per game how much mirror/translation symmetry to use (soft gated
symmetric attention bias + an equivariance-gap detector + symmetry-averaged search at the root); causal gated attention
over the imagined unroll replacing the LSTM value-prefix; re-closing GTrXL gates as a gentle plasticity reset; a
mechanistic probe study of the latent (ball/paddle, imagined-vs-real latent error, attention vs objects).

## What to search for

Search broadly (arXiv, OpenReview for ICLR/NeurIPS/ICML 2023-2026, TMLR, RLC, workshop papers, GitHub issues of
EfficientZero / EZ-V2 / LightZero / mctx) around these directions, and bring anything that suggests a **gap**:

1. Transformer **inductive biases for latent dynamics**: relative/rotary positions, locality priors, gating schedules,
   token pruning or merging of latent grids, attention over imagined vs real trajectories.
2. **Search x representation** interactions: using the world model's structure (symmetries, object slots, uncertainty,
   attention maps) inside Gumbel/MCTS search; transpositions and state abstraction in latent search; search-time
   ensembling or augmentation (test-time equivariance, test-time training of the dynamics).
3. **Value/reward target design** in the 100k regime: n-step and lambda targets under reanalyze, target staleness,
   reward-prefix vs per-step reward, distributional targets with the h-transform, raw-score objectives.
4. **Self-supervised consistency** beyond one-step SimSiam: multi-step, action-conditioned, masked latent modelling, and
   whether they interact with attention trunks differently than with CNNs.
5. **Stability and seed variance** of transformers in small-data RL: initialisation, gating, normalisation, learning-rate
   warmup, anything shown to shrink between-seed variance (this is our statistical bottleneck).
6. **Evaluation methodology** for Atari 100k: seed x episode variance, bimodal outcomes (Breakout's tunnel), how many
   seeds vs episodes (Neyman allocation), reporting practices.

## Output format

Return **8-12 candidate ideas**, ranked. For each:

1. **Name and one-sentence idea.**
2. **Mechanism:** why it should improve sample efficiency, stability or search quality in an EZ-V2-style agent.
3. **Closest prior work (2-4 papers)** with title, venue/year and arXiv ID or URL, and **exactly how the idea differs**.
4. **Novelty confidence** (high / medium / low) and what you searched to reach it (list the queries).
5. **Evidence it would work:** results from adjacent settings (model-free Atari, continuous control, vision) with numbers.
6. **Implementation sketch** in EZ-V2 terms (which module changes, roughly how many lines, extra compute per update).
7. **Minimal falsification experiment** within the budget: games, steps (10k / 30k / 100k), seeds, metric, and the result
   that would kill the idea.
8. **Risks and failure modes**, including whether it only works on one game.
9. **Probability of a reportable positive result** (your estimate, with one sentence of reasoning).

Then add:
- a short list of **ideas you considered and rejected because they are already published** (with the citation), so we do
  not rediscover them;
- **3 open questions** where the literature disagrees or is silent and a small controlled experiment would be a
  contribution by itself.

## Rules

- **Never invent citations.** Every paper must have a working arXiv ID, DOI or URL; if you cannot verify one, say so.
- Distinguish **"not found after searching X, Y, Z"** from **"does not exist"**.
- Prefer ideas testable on **several Atari 100k games**, not only Breakout; say which games would show the effect and which
  would serve as negative controls.
- Be concrete and quantitative; skip generic advice ("tune hyperparameters", "use more data").
