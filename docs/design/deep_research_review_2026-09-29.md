# Review of the deep-research report on Atari 100k novelty (2026-09-29)

Report produced from `deep_research_prompt_atari_novelty.md` (10 ranked proposals; the pasted copy was cut off inside the
"published ideas considered and rejected" section). Checked against sources and against our code on 2026-09-29.

## Errors found in the report

1. **Wrong attribution:** Zhai et al. 2023 (ICML, arXiv 2303.06296) proposed **sigma-Reparam** (spectral-normalised linear layers
   + a learned scalar), not "CosFormer".
2. **Evidence inverted (proposal 5):** the report says DiT's cross-attention conditioning improved FID by 34%. DiT
   (arXiv 2212.09748) found **adaLN-Zero better than cross-attention and in-context** at every stage, at lower compute.
3. **False premise about our code (proposal 3):** it assumes EZ-V2 average-pools the latent before SimSiam. Our
   `EZV2Model.project()` uses `s.flatten(1)` (the full 64x6x6 grid, no pooling).
4. **Missed prior work:** 2D RoPE for ViTs (RoPE-ViT, Heo et al., ECCV 2024, arXiv 2403.13298); masked latent prediction
   with action conditioning in RL (MLR, NeurIPS 2022, arXiv 2201.12096, Atari 100k; Masked Latent Transformers, arXiv 2507.04075);
   EfficientZero's own off-policy correction (adaptive horizon) and EZ-V2's search-based value estimation for proposal 7.
5. **Wrong ID:** "What model does MuZero learn?" is arXiv 2306.00840 (ECAI 2024 version exists), not 2407.01428.
6. **Unverified:** the V-MCTS ID 2305.20050; several "adjacent evidence" numbers (e.g. "42% less cross-seed variance",
   "Masked World Models +28% on Atari") have no traceable source. Success probabilities (50-85%) are uncalibrated.
7. **Generic paths:** the sketches name files we do not have; ours are `atari_qwen/models/ez_model.py`,
   `atari_qwen/models/gtrxl_layers.py`, `atari_qwen/training/train_ez_offpolicy.py`, `gumbel_mcts.py`.

## Verdict per proposal

| # | Proposal | Verdict | Where it went |
|---|---|---|---|
| 1 | 2D-axial RoPE in the GTrXL mixer | Keep as a cheap arm. Translation-relative but uses *signed* offsets, so it cannot express mirror symmetry; B26's offset bias can. Novelty only "in an MCTS latent dynamics model". | B31 (with B26) |
| 2 | Attention-head disagreement loss (Li et al. 2018, arXiv 1810.10183) | Keep with B18. Its "SD 11.3 -> < 6 over 6 seeds" test is underpowered; pre-register a variance test or use more seeds. | B32 |
| 3 | Patch-masked latent consistency | Drop: premise false for our code; MLR exists. | - |
| 4 | LSH latent transposition table in Gumbel search | Keep as an offline hit-rate probe first: with 16 simulations most expansions are depth 1-2, so hits may be rare. The report's own kill rule (hit rate < 5%) is right. | B33 |
| 5 | Action cross-attention in the dynamics | Merge into B29, conditioning via **adaLN-Zero** (per-action scale/shift, identity init) instead, per DiT. | B29 |
| 6 | Token merging (ToMe) in the unroll | Drop: 36-token attention is negligible compute (runs are launch-bound, B23); convs and heads need the full grid. | - |
| 7 | Retrace(lambda) targets under the h-transform in reanalyze | Keep as Later; weigh against EfficientZero's adaptive horizon and EZ-V2's SVE. | B34 |
| 8 | Attention-entropy adaptive simulations | Low: the mixer starts at identity (weak signal early); V-MCTS covers adaptive budgets. | - |
| 9 | Spectral-norm (contractive) value-prefix LSTM | Low: LSTM gates bound the state; 5-step unroll. | - |
| 10 | Test-time dynamics fine-tuning | Low: extra wall-clock, forgetting risk; the report gives it 50%. | - |

Sources checked: RoPE-ViT (arXiv 2403.13298, github naver-ai/rope-vit); What model does MuZero learn (arXiv 2306.00840);
sigma-Reparam (arXiv 2303.06296); DiT (arXiv 2212.09748, ICCV 2023); MLR (arXiv 2201.12096, NeurIPS 2022);
Masked Latent Transformers (arXiv 2507.04075).
