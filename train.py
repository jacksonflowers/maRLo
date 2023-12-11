from typing import Dict
import argparse
import numpy as np
from pyboy import PyBoy
from mario_env import MarioEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordVideo, TimeLimit
from custom_frame_stack import CustomFrameStack
from custom_record_video import CustomRecordVideo
from custom_monitor import CustomMonitor
from custom_cnn_minimal import CustomCNNMinimal

def make_env(args, rank, seed=0):
    def _init():
        env = MarioEnv(args)
        if args.observation_type == 'raw':
            env = GrayScaleObservation(env)
            env = ResizeObservation(env, (64, 80))
        env = CustomFrameStack(env, 4)
        env = TimeLimit(env, 4096)
        env = CustomRecordVideo(env, video_folder=f'videos/{args.model}_{args.skip_frames}_{args.observation_type}_{args.action_space}', name_prefix='')
        env = CustomMonitor(env, 'logs', name_prefix=f"{args.model}_{args.skip_frames}_{args.observation_type}_{args.action_space}")
        env.reset()
        return env
    
    set_random_seed(seed)
    return _init

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gb_path', type=str, default='roms/mario.gb')
    parser.add_argument('--num_cpu', type=int, default=16)
    parser.add_argument('--total_grad_updates', type=int, default=5)
    parser.add_argument('--observation_type', type=str, default='raw')
    parser.add_argument('--action_type', type=str, default='toggle')
    parser.add_argument('--action_space', type=str, default='all')
    parser.add_argument('--skip_frames', type=int, default=2)
    parser.add_argument('--model', type=str, default='ppo', choices=['ppo', 'dqn'])
    parser.add_argument('--checkpoint', action='store_true')

    args = parser.parse_args()
    print(args)

    vec_env = SubprocVecEnv([make_env(args, i) for i in range(args.num_cpu)])

    if args.checkpoint:
        save_freq = 4096 * max(args.total_grad_updates // 10, 1)
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"./logs/{args.model}_{args.skip_frames}_{args.observation_type}_{args.action_space}",
            name_prefix="",
        )

    policy_kwargs = None
    if args.observation_type == 'compressed':
        policy_kwargs = dict(
            features_extractor_class=CustomCNNMinimal,
        )

    if args.model == 'ppo':
        model = PPO('CnnPolicy', env=vec_env, verbose=1, n_epochs=3, batch_size=256, n_steps=4096, learning_rate=0.0002, vf_coef=1, ent_coef=0.01, policy_kwargs=policy_kwargs)
    elif args.model == 'dqn':
        model = DQN('CnnPolicy', env=vec_env, verbose=1, policy_kwargs=policy_kwargs)
    
    if args.checkpoint:
        model.learn(total_timesteps=args.num_cpu * 4096 * args.total_grad_updates, progress_bar=True, callback=checkpoint_callback)
    else:
        model.learn(total_timesteps=args.num_cpu * 4096 * args.total_grad_updates, progress_bar=True)

    model.save(f'{args.model}_{args.skip_frames}_{args.observation_type}_{args.action_space}')

if __name__ == '__main__':
    main()


# wall clock
# ppo_2_raw_all: 1hr
# ppo_2_compressed_all: 50min
# ppo_4_raw_all: 1hr 10min