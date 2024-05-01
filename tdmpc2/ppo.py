# Write a PPO agent for the ParkourEnv environment using stable baselines3.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import numpy as np
from envs.parkour import ParkourEnv
from common.parser import parse_cfg
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import hydra

class PPOParkourEnv(ParkourEnv):
    def __init__(self, cfg, exp_conf_path):
        super(PPOParkourEnv, self).__init__(cfg, exp_conf_path)
        self.max_episode_steps = 5
    
    def step(self, action):
        obs, reward, done, info = super(PPOParkourEnv, self).step(action)
        return obs, reward, done, False, info
    
    def reset(self, seed=0):
        return super(PPOParkourEnv, self).reset(), {}
    
    def get_reward(self, prev_pose, current_pose):
        return self.forward_reward_weight * np.exp(-0.15 * np.abs(current_pose[0] - 10))

@hydra.main(version_base=None, config_name='config', config_path='/home/learning/prashanth/tdmpc2/tdmpc2/')
def train_ppo(cfg: dict):
    cfg = parse_cfg(cfg)
    def make_env(cfg):
        env = PPOParkourEnv(cfg, cfg.exp_conf_path)
        return env
    # check_env(env)
    from functools import partial
    env_fn = partial(make_env, cfg)
    vec_env = make_vec_env(env_fn, n_envs=30, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
    ppo_args = {}
    ppo_args["batch_size"] = 64
    ppo_args["n_steps"] = 2048
    ppo_args["gamma"]= 0.95
    ppo_args["learning_rate"] = 3.56987e-05
    ppo_args["ent_coef"] = 0.00238306
    ppo_args["clip_range"] = 0.3
    ppo_args["n_epochs"] = 5
    ppo_args["gae_lambda"] = 0.9
    ppo_args["max_grad_norm"] = 2
    ppo_args["vf_coef"] = 0.431892
    ppo_args["policy_kwargs"] = dict(
                    log_std_init =-2,
                    ortho_init =False,
                    activation_fn =nn.Tanh,
                    net_arch =dict(pi =[128, 128], vf =[128, 128])
                    )
    model = PPO('MlpPolicy', vec_env, tensorboard_log='/home/learning/prashanth/tdmpc2/tdmpc2/sb3/logs/', verbose=0, **ppo_args)

    eval_callback = EvalCallback(eval_env, best_model_save_path='/home/learning/prashanth/tdmpc2/tdmpc2/sb3/', log_path='/home/learning/prashanth/tdmpc2/tdmpc2/sb3/', eval_freq=2049, deterministic=True)
    
    model.learn(total_timesteps=cfg.steps, progress_bar=True, callback=eval_callback)
    
    model.save('/home/learning/prashanth/tdmpc2/tdmpc2/sb3/final')

@hydra.main(version_base=None, config_name='config', config_path='/home/learning/prashanth/tdmpc2/tdmpc2/')
def eval_ppo(cfg: dict):
    cfg = parse_cfg(cfg)
    def make_env(cfg):
        env = PPOParkourEnv(cfg, cfg.exp_conf_path)
        return env
    from functools import partial
    env_fn = partial(make_env, cfg)
    # vec_env = make_vec_env(env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
    vec_env = env_fn()
    model = PPO.load('/home/learning/prashanth/tdmpc2/tdmpc2/best_model.zip')
    for _ in range(10):
        obs, _ = vec_env.reset()
        done = False
        vec_env.sim.viewer_paused = True
        vec_env.sim.viewer.update_hfield(0)
        while not done:
            if not vec_env.sim.viewer_paused:
                vec_env.sim.viewer.sync()
                # print(model.predict(obs))
                action, _states = model.predict(obs)
                obs, rewards, done, _, info = vec_env.step(action)
                print()
                # vec_env.render("human")
    vec_env.close()

if __name__ == '__main__':
    # cfg = {
    #     'env_name': 'ParkourEnv',
    #     'n_steps': 2048,
    #     'batch_size': 64,
    #     'lr': 3e-4,
    #     'total_timesteps': 100000,
    #     'save_path': 'ppo_parkour'
    # }
    train_ppo()
    # eval_ppo()

