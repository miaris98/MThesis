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
    env_type: str = "camera_easycarla"
    backbone: str = "resnet18"
    policy_arch: str = "qwen500m"
    fresh: bool = False
    weights_path: Optional[str] = None
    freeze_backbone: bool = True
    use_pretrained: bool = True
    
    total_steps: int = 50000
    rollout_steps: int = 500
    frame_skip: int = 2
    ppo_epochs: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.05
    minibatch_size: int = 128
    reward_clip: float = 50.0

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

    @classmethod
    def from_args(cls, args_list: Optional[List[str]] = None) -> "TrainingConfig":
        """Build TrainingConfig instance by parsing command-line flags or explicit argument list."""
        parser = argparse.ArgumentParser(description="Train PPO Deep RL Agent in CARLA Simulator.")
        parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
        parser.add_argument("--port", type=int, default=2000, help="CARLA port")
        parser.add_argument("--env-type", type=str, default="camera_easycarla", choices=["camera_easycarla", "carla_gym"])
        parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "lav", "erfnet", "qwen900m", "qwen500m", "qwen", "transformer"])
        parser.add_argument("--policy-arch", type=str, default="qwen500m", choices=["qwen500m", "qwen900m", "mlp", "transformer"])
        parser.add_argument("--fresh", action="store_true", default=False, help="Start training fresh from scratch")
        parser.add_argument("--weights-path", type=str, default=None, help="Path to custom pretrained checkpoint (.pth)")
        parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze vision backbone parameters")
        parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone")
        parser.add_argument("--use-pretrained", action="store_true", default=True, help="Use pretrained vision backbone")
        parser.add_argument("--no-pretrained", action="store_false", dest="use_pretrained")

        parser.add_argument("--total-steps", type=int, default=50000)
        parser.add_argument("--rollout-steps", type=int, default=500)
        parser.add_argument("--frame-skip", type=int, default=2)
        parser.add_argument("--ppo-epochs", type=int, default=4)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--gae-lambda", type=float, default=0.95)
        parser.add_argument("--clip-coef", type=float, default=0.2)
        parser.add_argument("--ent-coef", type=float, default=0.05)
        parser.add_argument("--minibatch-size", type=int, default=128)
        parser.add_argument("--reward-clip", type=float, default=50.0)

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
        
        return cls(
            host=parsed.host,
            port=parsed.port,
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
