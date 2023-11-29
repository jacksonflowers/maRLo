from typing import Dict
import argparse
import numpy as np
from pyboy import PyBoy
from mario_env import MarioEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation, RecordVideo, TimeLimit
from custom_frame_stack import CustomFrameStack
from custom_record_video import CustomRecordVideo

def make_env(args, rank, seed=0):
    def _init():
        env = MarioEnv(args)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, (72, 80))
        env = CustomFrameStack(env, 4)
        env = TimeLimit(env, 2048)
        env = CustomRecordVideo(env, video_folder='videos')
        env.reset()
        return env
    
    set_random_seed(seed)
    return _init

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gb_path', type=str, default='roms/mario.gb')
    parser.add_argument('--num_cpu', type=int, default=16)
    parser.add_argument('--total_grad_updates', type=int, default=5)
    parser.add_argument('--observation_type', type=str, default='compressed')
    parser.add_argument('--action_type', type=str, default='toggle')
    parser.add_argument('--model', type=str, default='ppo', choices=['ppo', 'dqn'])

    args = parser.parse_args()
    print(args)

    vec_env = SubprocVecEnv([make_env(args, i) for i in range(args.num_cpu)])

    if args.model == 'ppo':
        model = PPO('CnnPolicy', env=vec_env, verbose=1, n_epochs=5)
    elif args.model == 'dqn':
        model = DQN('CnnPolicy', env=vec_env, verbose=1)
    model.learn(total_timesteps=args.num_cpu * 2048 * args.total_grad_updates, progress_bar=True)

    model.save(f'{args.model}_mario')

    # obs = vec_env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = vec_env.step(action)
    #     print(done, rewards)
    #     if np.all(done):
    #         print(rewards)
    #         break

if __name__ == '__main__':
    main()