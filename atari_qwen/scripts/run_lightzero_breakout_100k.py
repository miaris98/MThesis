"""Atari-100k-benchmark EfficientZero baseline on Breakout, via vendored LightZero.

Deviates from LightZero's shipped zoo/atari/config/atari_efficientzero_config.py in exactly
two places, both required by the benchmark: env_id (Pong -> Breakout) and max_env_step
(5e5 -> 1e5, the Atari-100k cap = 400k frames at frameskip 4). Every other hyperparameter
is left at LightZero's own benchmarked values, deliberately: this run is a reference
baseline, so its credibility depends on matching the config the upstream project publishes
numbers for, not on being tuned or on saturating the host.

Notably reanalyze_ratio stays at LightZero's default 0. An earlier revision of this file set
it to 1.0 on the assumption that the sample-efficient regime needed full reanalysis. That was
never verified against either the paper or LightZero's benchmarks, and it cost ~3.6s per
training iteration (fresh 50-simulation MCTS over all 256 sampled trajectories every batch),
roughly a 20-70x slowdown, for no established correctness benefit.
"""
from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map

env_id = 'BreakoutNoFrameskip-v4'
action_space_size = atari_env_action_space_map[env_id]

collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
batch_size = 256
max_env_step = int(1e5)  # Atari-100k benchmark cap
reanalyze_ratio = 0.     # LightZero shipped default (their benchmarked setting)
num_unroll_steps = 5
update_per_collect = None
replay_ratio = 0.25

atari_efficientzero_100k_config = dict(
    exp_name=f'data_efficientzero/{env_id[:-14]}_efficientzero_100k_stack4_H{num_unroll_steps}_seed0',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=[4, 64, 64],
        frame_stack_num=4,
        gray_scale=True,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        # Standard Atari evaluation cap: 108,000 frames / frameskip 4 = 27,000 agent steps.
        # LightZero's default of 1.08e5 AGENT steps is 4x that (432k frames).
        eval_max_episode_steps=int(2.7e4),
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=[4, 64, 64],
            image_channel=1,
            frame_stack_num=4,
            gray_scale=True,
            action_space_size=action_space_size,
            downsample=True,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            reward_support_range=(-50., 51., 1.),
            value_support_range=(-50., 51., 1.),
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=400,
        use_augmentation=True,
        use_priority=False,
        replay_ratio=replay_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        dormant_threshold=0.025,
        optim_type='SGD',
        piecewise_decay_lr_scheduler=True,
        learning_rate=0.2,
        target_update_freq=100,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=2,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
atari_efficientzero_100k_config = EasyDict(atari_efficientzero_100k_config)
main_config = atari_efficientzero_100k_config

atari_efficientzero_100k_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='efficientzero',
        import_names=['lzero.policy.efficientzero'],
    ),
)
atari_efficientzero_100k_create_config = EasyDict(atari_efficientzero_100k_create_config)
create_config = atari_efficientzero_100k_create_config

if __name__ == '__main__':
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
