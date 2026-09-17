"""Configuration dataclass and presets for Qwen Atari agent."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

QWEN_ATARI_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {
        "depth": 4,
        "embed_dim": 256,
        "num_heads": 4,
        "ffn_dim": 1024,
        "approx_params": "3.5M",
        "description": "Ultra-fast prototyping for rapid iterations and reward convergence"
    },
    "small": {
        "depth": 8,
        "embed_dim": 512,
        "num_heads": 8,
        "ffn_dim": 2048,
        "approx_params": "25M",
        "description": "High-capacity transformer balanced for single-GPU throughput"
    },
    "100m": {
        "depth": 12,
        "embed_dim": 768,
        "num_heads": 12,
        "ffn_dim": 2816,
        "approx_params": "106M",
        "description": "Direct match to Qwen-100M preset from src.models.qwen_sac_heads"
    },
    "500m": {
        "depth": 28,
        "embed_dim": 1024,
        "num_heads": 16,
        "ffn_dim": 4096,
        "approx_params": "490M",
        "description": "Large-scale transformer for scaling study comparisons"
    }
}


@dataclass
class AtariConfig:
    """Master configuration for Atari Qwen training and evaluation."""
    
    # Environment
    env_id: str = "BreakoutNoFrameskip-v4"
    frame_stack: int = 4
    frame_size: int = 84
    episodic_life: bool = True
    reward_clipping: bool = True
    noop_max: int = 30
    
    # Architecture
    model_preset: str = "tiny"           # "tiny", "small", "100m", "500m"
    encoder_type: str = "nature_cnn"      # "nature_cnn", "impala_cnn", "patch"
    patch_size: int = 14                 # Used if encoder_type == "patch" (84 / 14 = 6x6 patches)
    features_dim: int = 512
    dropout: float = 0.0
    use_gradient_checkpointing: bool = False  # Enable for 100m/500m if VRAM constrained
    
    # PPO Rollout & Optimization
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    lr_schedule: str = "cosine"          # "cosine", "linear", "constant"
    warmup_steps: int = 1000
    num_envs: int = 16
    num_steps: int = 128                 # Batch per rollout = num_envs * num_steps = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4             # Minibatch size = 2048 / 4 = 512
    update_epochs: int = 4
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    ent_coef_end: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    
    # Hardware & Performance
    use_amp: bool = True                 # PyTorch Automatic Mixed Precision (fp16/bf16)
    amp_dtype: str = "bfloat16"          # "float16" or "bfloat16" (Ada Lovelace RTX 40xx has fast bf16)
    compile_model: bool = False          # torch.compile (can be toggled)
    seed: int = 42
    
    # Logging & Checkpoints
    exp_name: str = "atari_qwen_ppo"
    log_dir: str = "results/atari_qwen"
    save_freq_steps: int = 100_000
    eval_freq_steps: int = 50_000
    eval_episodes: int = 10
    track_mlflow: bool = True
    mlflow_tracking_uri: Optional[str] = None
    
    def get_preset_kwargs(self) -> Dict[str, Any]:
        """Return preset kwargs (depth, embed_dim, num_heads, ffn_dim)."""
        key = self.model_preset.lower()
        for k in ("500m", "100m", "small", "tiny"):
            if k in key:
                return {
                    "depth": QWEN_ATARI_PRESETS[k]["depth"],
                    "embed_dim": QWEN_ATARI_PRESETS[k]["embed_dim"],
                    "num_heads": QWEN_ATARI_PRESETS[k]["num_heads"],
                    "ffn_dim": QWEN_ATARI_PRESETS[k]["ffn_dim"],
                }
        return {
            "depth": QWEN_ATARI_PRESETS["tiny"]["depth"],
            "embed_dim": QWEN_ATARI_PRESETS["tiny"]["embed_dim"],
            "num_heads": QWEN_ATARI_PRESETS["tiny"]["num_heads"],
            "ffn_dim": QWEN_ATARI_PRESETS["tiny"]["ffn_dim"],
        }


def get_config(**overrides) -> AtariConfig:
    """Instantiate an AtariConfig with optional custom field overrides."""
    return AtariConfig(**overrides)
