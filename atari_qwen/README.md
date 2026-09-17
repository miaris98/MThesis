# Atari Qwen: Transformer Reinforcement Learning for Atari 2600

Applying Qwen LLM/VLM transformer architecture components (`RMSNorm`, `SwiGLU`, and `QwenTransformerBlock` with trainable alpha gating) to discrete control benchmarks on the **Arcade Learning Environment (Atari 2600)** to compete with State-of-the-Art (SOTA) baselines (DQN, Rainbow, PPO, Agent57).

---

## 1. Architectural Highlights

- **Shared Transformer Trunk**: Directly imports `RMSNorm`, `SwiGLU`, and `QwenTransformerBlock` from `src.models.transformer.layers`.
- **Spatial Self-Attention**: Rather than crushing frames into a flat vector, visual tokenizers (Nature CNN, Impala CNN, or Patch Tokenizer) produce spatial token grids (e.g. $7 \times 7 = 49$ or $11 \times 11 = 121$ tokens), allowing multi-head self-attention to relate dynamic game entities (ball, paddle, enemies) across the screen.
- **Dual Query Tokens**: Dedicated learnable `[ACTOR]` and `[CRITIC]` tokens query the visual sequence in parallel.
- **Trainable Skip Gating**: Qwen alpha parameters (`alpha_attn`, `alpha_ffn`) adaptively balance shallow vs. deep representations throughout training.
- **Mixed Precision**: Automatic Mixed Precision (`bfloat16` / `float16`) for maximum throughput on modern NVIDIA tensor cores.

---

## 2. Model Presets

| Preset | Layers | Hidden Dim | Heads | FFN Dim | Approx Params | Best Used For |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`tiny`** | 4 | 256 | 4 | 1024 | ~3.5M | Rapid iteration, fast reward convergence |
| **`small`** | 8 | 512 | 8 | 2048 | ~25M | High-capacity balance for single GPU |
| **`100m`** | 12 | 768 | 12 | 2816 | ~106M | SOTA comparison scaling study |
| **`500m`** | 28 | 1024 | 16 | 4096 | ~490M | Multi-GPU / massive compute scaling |

---

## 3. Vast.ai Quickstart

On your rented Vast.ai instance (RTX 4070 Ti Super / RTX 3090 / RTX 4090):

```bash
# 1. Setup environment (Gymnasium, ALE-Py, AutoROM, OpenCV, etc.)
bash atari_qwen/scripts/setup_vastai_atari.sh

# 2. Launch training run
bash atari_qwen/scripts/run_experiment.sh BreakoutNoFrameskip-v4 tiny 16 10000000
```

---

## 4. Evaluation & Human-Normalized Score (HNS)

To evaluate a trained checkpoint against standard DeepMind / Human baselines:

```bash
python atari_qwen/eval/evaluate.py \
    --checkpoint results/atari_qwen/<run_name>/checkpoints/model_best.pt \
    --env-id BreakoutNoFrameskip-v4 \
    --num-episodes 30
```

$$\text{HNS} = \frac{\text{Score}_{\text{agent}} - \text{Score}_{\text{random}}}{\text{Score}_{\text{human}} - \text{Score}_{\text{random}}} \times 100\%$$
