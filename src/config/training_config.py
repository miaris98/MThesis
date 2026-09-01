"""Training configuration dataclasses and command-line argument parser."""
import os
import argparse
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for PPO training in CARLA simulator."""
    host: str = "127.0.0.1"
    port: int = 2000
    num_envs: int = 2
    carla_ports: Optional[List[int]] = None
    env_type: str = "camera_easycarla"
    backbone: str = "resnet18"
    policy_arch: str = "qwen100m"
    fresh: bool = False
    weights_path: Optional[str] = None
    freeze_backbone: bool = True
    use_pretrained: bool = True
    
    total_steps: int = 70000
    rollout_steps: int = 500
    frame_skip: int = 2
    ppo_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    minibatch_size: int = 128
    reward_clip: float = 50.0

    # Algorithm selection and off-policy (SAC) hyperparameters
    reward_fn: str = "custom_1"
    algo: str = "ppo"
    sac_policy_arch: str = "mlp"
    buffer_size: int = 100000
    sac_batch_size: int = 256
    tau: float = 0.005
    learning_starts: int = 1000
    updates_per_step: int = 1
    init_alpha: float = 0.2
    autotune_alpha: bool = True

    # Early stopping based on evaluation / moving average episode performance
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1.0
    early_stopping_window: int = 10
    target_reward: Optional[float] = None

    log_dir: str = "/workspace/runs"
    checkpoint_dir: str = "/workspace/checkpoints"
    resume: bool = False
    
    num_vehicles: int = 3
    num_walkers: int = 10
    town: str = "Town10HD_Opt"
    
    use_mlflow: bool = True
    experiment_name: str = "CARLA_PPO_RL"
    mlflow_port: int = 10100
    compile: bool = False

    def get_ports(self) -> List[int]:
        """Return list of active CARLA ports for parallel environments."""
        if self.carla_ports and len(self.carla_ports) > 0:
            return self.carla_ports
        # CARLA uses 3 consecutive ports per instance (P, P+1, P+2). Stride by 4 to prevent collisions.
        return [self.port + (i * 4) for i in range(self.num_envs)]

    @classmethod
    def from_args(cls, args_list: Optional[List[str]] = None) -> "TrainingConfig":
        """Build TrainingConfig instance by parsing command-line flags or explicit argument list."""
        parser = argparse.ArgumentParser(description="Train PPO Deep RL Agent in CARLA Simulator.")
        parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
        parser.add_argument("--port", type=int, default=2000, help="CARLA primary port")
        parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel CARLA server environments")
        parser.add_argument("--carla-ports", type=str, default=None, help="Comma-separated list of CARLA server ports (e.g. 2000,2004)")
        parser.add_argument("--env-type", type=str, default="camera_easycarla", choices=["camera_easycarla", "carla_gym"])
        parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "lav", "erfnet", "qwen100m", "qwen500m", "qwen900m", "qwen", "transformer"])
        parser.add_argument("--policy-arch", type=str, default="qwen100m", choices=["qwen100m", "qwen500m", "qwen900m", "mlp", "transformer"])
        parser.add_argument("--fresh", action="store_true", default=False, help="Start training fresh from scratch")
        parser.add_argument("--weights-path", type=str, default=None, help="Path to custom pretrained checkpoint (.pth)")
        parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze vision backbone parameters")
        parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone")
        parser.add_argument("--use-pretrained", action="store_true", default=True, help="Use pretrained vision backbone")
        parser.add_argument("--no-pretrained", action="store_false", dest="use_pretrained")

        parser.add_argument("--total-steps", type=int, default=70000)
        parser.add_argument("--rollout-steps", type=int, default=500)
        parser.add_argument("--frame-skip", type=int, default=2)
        parser.add_argument("--ppo-epochs", type=int, default=4)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--gae-lambda", type=float, default=0.95)
        parser.add_argument("--clip-coef", type=float, default=0.2)
        parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy bonus; must stay well below the reward scale")
        parser.add_argument("--minibatch-size", type=int, default=128)
        parser.add_argument("--reward-clip", type=float, default=50.0)

        # Algorithm selection and SAC-specific flags
        parser.add_argument("--reward-fn", type=str, default="custom_1", choices=["custom_1", "leaderboard", "roach", "interp_e2e"], help="Swappable reward function")
        parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="Training algorithm")
        parser.add_argument("--sac-policy-arch", type=str, default="mlp", choices=["mlp", "qwen100m", "qwen500m", "qwen900m"], help="SAC actor/critic architecture")
        parser.add_argument("--buffer-size", type=int, default=100000, help="SAC replay buffer capacity")
        parser.add_argument("--sac-batch-size", type=int, default=256, help="SAC gradient minibatch size")
        parser.add_argument("--tau", type=float, default=0.005, help="SAC target network Polyak coefficient")
        parser.add_argument("--learning-starts", type=int, default=1000, help="Random-action steps before SAC updates begin")
        parser.add_argument("--updates-per-step", type=int, default=1, help="SAC gradient steps per environment step")
        parser.add_argument("--init-alpha", type=float, default=0.2, help="Initial SAC entropy temperature")
        parser.add_argument("--no-autotune-alpha", action="store_false", dest="autotune_alpha", help="Freeze the entropy temperature")

        # Early stopping flags
        parser.add_argument("--early-stopping", action="store_true", default=True, help="Enable performance-based early stopping")
        parser.add_argument("--no-early-stopping", action="store_false", dest="early_stopping", help="Disable early stopping")
        parser.add_argument("--early-stopping-patience", "--patience", type=int, default=20, help="Patience rollouts without improvement")
        parser.add_argument("--early-stopping-min-delta", "--min-delta", type=float, default=1.0, help="Minimum delta to qualify as improvement")
        parser.add_argument("--early-stopping-window", type=int, default=10, help="Window size for moving average reward")
        parser.add_argument("--target-reward", type=float, default=None, help="Target reward threshold for immediate success stop")

        parser.add_argument("--log-dir", type=str, default="/workspace/runs")
        parser.add_argument("--checkpoint-dir", type=str, default="/workspace/checkpoints")
        parser.add_argument("--resume", action="store_true", default=False)
        parser.add_argument("--num-vehicles", type=int, default=3)
        parser.add_argument("--num-walkers", type=int, default=10)
        parser.add_argument("--town", type=str, default="Town10HD_Opt")

        parser.add_argument("--use-mlflow", action="store_true", default=True)
        parser.add_argument("--no-mlflow", action="store_false", dest="use_mlflow")
        parser.add_argument("--experiment-name", type=str, default="CARLA_PPO_RL")
        parser.add_argument("--mlflow-port", type=int, default=10100)
        parser.add_argument("--compile", action="store_true", default=False)

        parsed = parser.parse_args(args_list) if args_list is not None else parser.parse_args()

        ports_list = None
        if parsed.carla_ports:
            ports_list = [int(p.strip()) for p in parsed.carla_ports.split(",") if p.strip()]
            num_envs = len(ports_list)
        else:
            num_envs = parsed.num_envs
        
        return cls(
            host=parsed.host,
            port=parsed.port,
            num_envs=num_envs,
            carla_ports=ports_list,
            env_type=parsed.env_type,
            backbone=parsed.backbone,
            policy_arch=parsed.policy_arch,
            fresh=parsed.fresh,
            weights_path=parsed.weights_path,
            freeze_backbone=parsed.freeze_backbone,
            use_pretrained=parsed.use_pretrained,
            total_steps=parsed.total_steps,
            rollout_steps=parsed.rollout_steps,
            frame_skip=parsed.frame_skip,
            ppo_epochs=parsed.ppo_epochs,
            lr=parsed.lr,
            gamma=parsed.gamma,
            gae_lambda=parsed.gae_lambda,
            clip_coef=parsed.clip_coef,
            ent_coef=parsed.ent_coef,
            minibatch_size=parsed.minibatch_size,
            reward_clip=parsed.reward_clip,
            reward_fn=parsed.reward_fn,
            algo=parsed.algo,
            sac_policy_arch=parsed.sac_policy_arch,
            buffer_size=parsed.buffer_size,
            sac_batch_size=parsed.sac_batch_size,
            tau=parsed.tau,
            learning_starts=parsed.learning_starts,
            updates_per_step=parsed.updates_per_step,
            init_alpha=parsed.init_alpha,
            autotune_alpha=parsed.autotune_alpha,
            early_stopping=parsed.early_stopping,
            early_stopping_patience=parsed.early_stopping_patience,
            early_stopping_min_delta=parsed.early_stopping_min_delta,
            early_stopping_window=parsed.early_stopping_window,
            target_reward=parsed.target_reward,
            log_dir=parsed.log_dir,
            checkpoint_dir=parsed.checkpoint_dir,
            resume=parsed.resume,
            num_vehicles=parsed.num_vehicles,
            num_walkers=parsed.num_walkers,
            town=parsed.town,
            use_mlflow=parsed.use_mlflow,
            experiment_name=parsed.experiment_name,
            mlflow_port=parsed.mlflow_port,
            compile=parsed.compile
        )
