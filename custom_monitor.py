import gymnasium as gym
import os
import csv

class CustomMonitor(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env, log_dir, model):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            log_dir=log_dir,
        )
        gym.Wrapper.__init__(self, env)

        os.makedirs(os.path.abspath(log_dir), exist_ok=True)
        log_file = os.path.join(log_dir, f'log_{model}.csv')
        self.file_handler = open(log_file, "at", newline="\n")
        self.logger = csv.DictWriter(
            self.file_handler, fieldnames=("episode", "step", "episode_reward", "world", "level", "level_progress", "episode_length")
        )
        self.file_handler.flush()
        
        self.env = env

        self.episode_id = 0
        self.step_id = 0
        self.rewards = []
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rewards = []
        return obs, info

    def step(self, action):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        self.rewards.append(reward)
        if terminated or truncated:
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            world, level = info['world']
            level_progress = info['level_progress']

            self.logger.writerow({
                'episode': self.episode_id,
                'step': self.step_id,
                'episode_reward': ep_rew,
                'world': world,
                'level': level,
                'level_progress': level_progress,
                'episode_length': ep_len,
            })
            self.file_handler.flush()
            
            self.episode_id += 1

        self.step_id += 1

        return observation, reward, terminated, truncated, info

    def close(self):
        super().close()
        self.file_handler.close()
