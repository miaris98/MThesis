# DeepSeek-V3 / V3.2 attention upgrades: what we can use (2026-10-10, S-139)

Sources read in full text: DeepSeek-V3 technical report (arXiv 2412.19437, Sections 2.1-2.2, 4.2, 4.5) and DeepSeek-V3.2 (arXiv 2512.02556, Section 2). Everything marked *paper* is quoted or read from those PDFs; *my arithmetic* and *my judgement* are ours.

## 1. What DeepSeek changed

**Multi-head Latent Attention (MLA)** (paper, V3 Sec. 2.1.1, eqs. 1-11). Keys and values are produced from one low-rank latent: `c_t = W_DKV h_t` (dimension d_c = 512), `k = W_UK c_t`, `v = W_UV c_t`; a separate decoupled key `k_R = RoPE(W_KR h_t)` (d_h^R = 64 per head) carries position because RoPE does not commute with the low-rank split. Queries are also compressed (d_c' = 1536), but only to save training activation memory. At inference only `c_t` and `k_R` are cached. V3 uses 61 layers, hidden size 7168, 128 heads of 128 dims. *My arithmetic:* a standard multi-head cache holds 2 x 128 x 128 = 32,768 numbers per token and layer, MLA holds 512 + 64 = 576, a 57x reduction. The paper states only that the cache is "significantly reduced" at MHA-level quality. Extra RMSNorm layers follow the compressed latents and scaling factors sit at the width bottlenecks (Sec. 4.2).

**DeepSeek Sparse Attention (DSA)** (paper, V3.2 Sec. 2.1). Added to a dense MLA model through continued training; the only architectural change from V3.1-Terminus.
- *Lightning indexer:* `I_{t,s} = sum_j w_{t,j} ReLU(q_{t,j} . k_s)` over a few indexer heads, run in FP8; scores every earlier token for each query.
- *Selection:* attention is computed only over the top-k = 2048 key-value entries, so core attention goes from O(L^2) to O(Lk); the indexer itself stays O(L^2) but is much cheaper.
- *Instantiated in MLA's multi-query mode* (one latent shared by all query heads), because the kernels need each key-value entry shared across queries.
- *Training:* (1) dense warm-up: freeze everything except the indexer, train it with a KL divergence to the head-summed, L1-normalised dense attention distribution, lr 1e-3, 1,000 steps x 16 sequences x 128K tokens = 2.1B tokens; (2) sparse stage: train all parameters, the indexer keeps the KL loss (on the selected set, indexer input detached so the main model only sees its language-model loss), lr 7.3e-6, 15,000 steps x 480 x 128K = 943.7B tokens. Result: no substantial quality loss against V3.1-Terminus on short and long context, large long-context cost reduction.

**Multi-Token Prediction (MTP)** (paper, V3 Sec. 2.2). D = 1 extra module: shared embedding and output head, one transformer block, a projection `M_k` (d x 2d) that merges the previous depth's hidden state with the embedding of the next token (eq. 21); the causal chain is kept (unlike parallel heads). Loss `lambda/D sum_k CE_k` (eq. 25). Discarded at inference (or reused for speculative decoding, second-token acceptance 85-90%). Ablation (Table 4): improves most benchmarks at 15.7B and 228.7B parameters.

**Auxiliary-loss-free expert balancing** (paper, V3 Sec. 2.1.2, eq. 16). A per-expert bias `b_i` is added to the affinity only when choosing the top-K experts; the gate value still uses the unbiased affinity. After each step `b_i` goes down by gamma if the expert was overloaded, up if underloaded. Ablation (Table 5): small gains over auxiliary-loss balancing, e.g. Pile BPB 0.727 -> 0.724 and 0.656 -> 0.652, BBH 37.3 -> 39.3 and 66.7 -> 67.9, HumanEval (large) 40.2 -> 46.3.

Not relevant to us: FP8 training, DualPipe, expert-parallel communication, node-limited routing, long-context RL infrastructure.

## 2. What our models look like (the constraint that decides everything)

CARLA head: frozen TF++ RegNetY-032 features pooled to 4 x 4 = 16 vision tokens plus 4 state tokens, **20 tokens per step**, one forward pass per control step, no autoregressive decoding, no cache. Atari: GTrXL token mixer on a single frame's feature map inside an EfficientZero network (96 px input, 5 unroll steps). At these lengths attention costs almost nothing and KV caching does not exist as a problem. MLA and DSA solve a long-sequence, long-decode problem we do not have.

## 3. Verdicts

| idea | verdict for our work | reason / when it would matter |
|---|---|---|
| MLA latent KV compression | **no use now** | no cache, 20 tokens. Would matter only if we add a streaming history (TODO A48: previous frames as tokens, e.g. 8 frames x 216 un-pooled tokens ~ 1,700 tokens cached at every control step): then a 57x smaller cache is real |
| DSA / lightning indexer | **no use now; one idea worth keeping** | no compute problem at 20 or even 216 tokens. The transferable idea is *supervising a cheap selector with the dense attention distribution (KL)*: it could choose which of the 216 un-pooled RegNet tokens (9 x 24) or which history frames the head reads, instead of the fixed 4 x 4 average pool that blurs a lead car (S-122). Only relevant after A54 (space-to-depth tokens) shows that finer tokens help at all |
| **MTP** | **cheapest real candidate** | a sequential auxiliary head predicting further-ahead waypoints (1.5-2.5 s) from the same trunk, trained only (dropped at inference), densifies supervision. It is the same family as TODO A53 (2.5 s horizon, scale-free loss). Caveat: our head-only changes have not beaten E15 (D2: A15, A38, A39 all inside seed noise ~4 DS), so expected gain is small; run it as one more head-only screen, not as the plan |
| Aux-loss-free bias balancing | **useful only if we build a mixture** | the multi-anchor / mixture head (TODO A41, A55, A23) has the usual failure of anchors dying; adding a per-anchor bias updated by usage is a one-line, loss-free fix. Do not build a mixture just for this |
| Bottleneck RMSNorm + scaled init (std 0.006) | **cheap hygiene** | matches the stability lessons from the GTrXL mixer (S-129, S-135); use the same recipe when adding any new bottleneck or latent |
| GRPO-style group-relative RL (V3 post-training, not an attention change) | **note for A20** | closed-loop RL fine-tuning of the driving policy could use group-relative advantages from several rollouts of the same route without a value network; the sim is the limit (~20 route-runs/h on a slow box), not the algorithm |

**Bottom line (my judgement):** the attention upgrades themselves buy us nothing at 20 tokens; the transferable parts are the training ideas (MTP as dense auxiliary supervision, KL-supervised selectors, bias-based balancing). None of them targets what limits our driving score (obstacle / passing behaviour and the vision bottleneck), so they rank below A54 and the Epic re-baseline (A61).

## 4. If we want to try the cheap one

MTP-style auxiliary waypoint head: add a second small transformer block after the policy token that takes the first 5 predicted waypoint embeddings plus the embedding of the route point beyond them (eq. 21 analogue) and predicts waypoints 6-10; loss weight lambda 0.3; evaluate on the pooled-token cache (A59) with the J-cache control, 2 seeds, as in D2. Expected cost: one head-only arm, ~15 min training per seed on a cached 3090, 1.1 box-hours per screen.
