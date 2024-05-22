# Write a PPO agent for the ParkourEnv environment using stable baselines3.

import torch.nn as nn
import numpy as np
from envs.parkour import ParkourEnv
from envs.parkour_dynamic import ParkourDynamic
from common.parser import parse_cfg
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import hydra
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os
PROJECT_PATH = os.path.dirname(__file__)

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
    
class PPOParkourDynamicEnv(ParkourDynamic):
    def __init__(self, cfg, exp_conf_path):
        super(PPOParkourDynamicEnv, self).__init__(cfg, exp_conf_path)
    
    def step(self, action):
        obs, reward, done, info = super(PPOParkourDynamicEnv, self).step(action)
        return obs, reward, done, False, info
    
    def reset(self, seed=0):
        return super(PPOParkourDynamicEnv, self).reset(), {}
    
    def get_reward(self, prev_pose, current_pose):
        return self.forward_reward_weight * np.exp(-0.15 * np.abs(current_pose[0] - 10))

@hydra.main(version_base=None, config_name='config', config_path=PROJECT_PATH)
def train_ppo(cfg: dict):
    cfg = parse_cfg(cfg)
    def make_env(cfg):
        env = PPOParkourEnv(cfg, cfg.exp_conf_path)
        return env
    # check_env(env)
    from functools import partial
    env_fn = partial(make_env, cfg)
    vec_env = make_vec_env(env_fn, n_envs=2, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
    ppo_args = {}
    ppo_args["batch_size"] = 64
    ppo_args["n_steps"] = 128
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
    model = PPO('MlpPolicy', vec_env, tensorboard_log='sb3/logs/', verbose=0, **ppo_args)

    eval_callback = EvalCallback(eval_env, best_model_save_path='sb3/tmp', log_path='sb3/tmp', eval_freq=2049, deterministic=True)
    
    model.learn(total_timesteps=cfg.steps, progress_bar=True, callback=eval_callback)
    
    model.save('sb3/finaltmp')

@hydra.main(version_base=None, config_name='config_dynamic', config_path=PROJECT_PATH)
def eval_ppo(cfg: dict):
    cfg = parse_cfg(cfg)
    def make_env(cfg):
        env = PPOParkourDynamicEnv(cfg, cfg.exp_conf_path)
        return env
    from functools import partial
    env_fn = partial(make_env, cfg)
    vec_env = env_fn()
    model = PPO.load('sb3/hp/50_64_32/best_model.zip')
    render = vec_env.render_viewer
    if render:
        vec_env.sim.viewer.cam.lookat = [4.05, -1.5, 0]
        vec_env.sim.viewer.cam.distance = 2.5
        vec_env.sim.viewer.cam.elevation = -20
        vec_env.sim.viewer.cam.azimuth = 135
    all_returns = []
    all_episode_lengths = []
    all_final_x = []
    for _ in range(20):
        obs, _ = vec_env.reset()
        done = False
        if render:
            vec_env.sim.viewer_paused = True
            vec_env.sim.viewer.update_hfield(0)
        returns = 0
        steps = 0
        while not done:
            if render and not vec_env.sim.viewer_paused:
                vec_env.sim.viewer.sync()
            # print(model.predict(obs))
            action, _states = model.predict(obs)
            # action = np.random.normal(0, 0.05, 2).clip(-1, 1)
        #action = np.random.uniform(-1, 1, 2)
            obs, rewards, done, _, info = vec_env.step(action)
            returns += rewards
            steps += 1
        print(f'Episode finished after {steps} steps with return {returns:.2f}. Final x = {vec_env.sim.data.qpos[0]:.2f}')
        all_returns.append(returns)
        all_episode_lengths.append(steps)
        all_final_x.append(vec_env.sim.data.qpos[0])
    vec_env.close()
    print(f'Mean return: {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}')
    print(f'Mean episode length: {np.mean(all_episode_lengths):.2f} +/- {np.std(all_episode_lengths):.2f}')
    print(f'Mean final x: {np.mean(all_final_x):.2f} +/- {np.std(all_final_x):.2f}')

@hydra.main(version_base=None, config_name='config_dynamic', config_path=PROJECT_PATH)
def hyper_param_search(cfg: dict):
    cfg = parse_cfg(cfg)
    cfg.steps = 122_880
    # cfg.steps = 600
    def make_env(cfg):
        env = PPOParkourDynamicEnv(cfg, cfg.exp_conf_path)
        return env
    from functools import partial
    env_fn = partial(make_env, cfg)
    ppo_args = {}
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
    # n_envs = [16, 24, 30]
    # batch_sizes = [32, 64, 128, 256, 512]
    # n_steps = [256, 512, 1024, 2048]
    n_envs = [50]
    batch_sizes = [64]
    n_steps = [32, 128, 512, 2048]
    for n_env in n_envs:
        for batch_size in batch_sizes:
            for n_step in n_steps:
                print("-------------------")
                print(f"n_env: {n_env}, batch_size: {batch_size}, n_step: {n_step}")
                ppo_args["batch_size"] = batch_size
                ppo_args["n_steps"] = n_step
                vec_env = make_vec_env(env_fn, n_envs=n_env, vec_env_cls=SubprocVecEnv)
                eval_env = make_vec_env(env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
                model = PPO('MlpPolicy', vec_env, tensorboard_log=f'sb3/hp/{n_env}_{batch_size}_{n_step}', verbose=0, **ppo_args)

                eval_callback = EvalCallback(eval_env, best_model_save_path='sb3/hp', log_path='sb3/hp', eval_freq=n_step+1, deterministic=True)
                
                model.learn(total_timesteps=cfg.steps, progress_bar=True, callback=eval_callback)
                
                model.save('sb3/hp/final')
                print("-------------------")

if __name__ == '__main__':
    # train_ppo()
    eval_ppo()
    # hyper_param_search()
