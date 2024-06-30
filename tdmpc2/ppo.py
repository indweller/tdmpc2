import torch.nn as nn
import numpy as np
from envs.parkour import ParkourEnv
from envs.parkour_dynamic import ParkourDynamic
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import hydra
from omegaconf import OmegaConf
from functools import partial
import warnings
warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt
PROJECT_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PROJECT_PATH, 'dtsd/confs/')

class PPOParkourEnv(ParkourEnv):
    def __init__(self, cfg):
        super(PPOParkourEnv, self).__init__(cfg)
        self.max_episode_steps = 5
    
    def step(self, action):
        obs, reward, done, info = super(PPOParkourEnv, self).step(action)
        return obs, reward, done, False, info
    
    def reset(self, seed=0):
        return super(PPOParkourEnv, self).reset(), {}
    
class PPOParkourDynamicEnv(ParkourDynamic):
    def __init__(self, cfg):
        super(PPOParkourDynamicEnv, self).__init__(cfg)
    
    def step(self, action):
        obs, reward, done, info = super(PPOParkourDynamicEnv, self).step(action)
        return obs, reward, done, False, info
    
    def reset(self, seed=0):
        return super(PPOParkourDynamicEnv, self).reset(), {}

def make_env(cfg):
    env = PPOParkourDynamicEnv(cfg)
    return env

def train_ppo(cfg):
    env_fn = partial(make_env, cfg)
    vec_env = make_vec_env(env_fn, n_envs=12, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
    ppo_args = {
        "batch_size": 64,
        "n_steps": 32,
        "gamma": 0.95,
        "learning_rate": 3.56987e-05,
        "ent_coef": 0.00238306,
        "clip_range": 0.3,
        "n_epochs": 5,
        "gae_lambda": 0.9,
        "max_grad_norm": 2,
        "vf_coef": 0.431892,
        "policy_kwargs": dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )
    }

    model = PPO('MlpPolicy', vec_env, tensorboard_log=f'sb3/logs/{cfg["exp_name"]}', verbose=0, **ppo_args)

    eval_callback = EvalCallback(eval_env, best_model_save_path=f'sb3/{cfg["exp_name"]}', log_path=f'sb3/{cfg["exp_name"]}', eval_freq=ppo_args["n_steps"] + 1, deterministic=True)
    
    model.learn(total_timesteps=cfg["steps"], progress_bar=True, callback=eval_callback)
    
    model.save(f'sb3/{cfg["exp_name"]}')

def eval_model(cfg, algo="PPO", episodes=50):
    env_fn = partial(make_env, cfg)
    vec_env = env_fn()
    if algo == "PPO":
        print("Loading PPO model")
        # model = PPO.load('sb3/hp/50_64_32/best_model.zip')
        model = PPO.load('../../discovery/sb3/high_oracle_freq/best_model.zip')
    else:
        model = SAC.load('../../discovery/sb3/sac_high_oracle_freq/best_model.zip')
    render = vec_env.render_viewer
    vec_env.sim.viewer_paused = False
    if render:
        vec_env.sim.viewer.cam.lookat = np.array([4.05, 0, 0])
        vec_env.sim.viewer.cam.azimuth = 90
        vec_env.sim.viewer.cam.elevation = -10
        vec_env.sim.viewer.cam.distance = 2.25
    all_returns = []
    all_episode_lengths = []
    all_final_x = []
    all_times = []
    all_modes = []
    for episode_no in range(episodes):
        obs, _ = vec_env.reset()
        done = False
        if render:
            # vec_env.sim.viewer_paused = True
            vec_env.sim.viewer.update_hfield(0)
        returns = 0
        steps = 0
        modes = []
        import time
        st = time.time()
        while not done:
            if render and not vec_env.sim.viewer_paused:
                # print(model.predict(obs))
                action, _states = model.predict(obs, deterministic=True)
                modes.append(vec_env.scale_actions(action))
                # action = np.random.normal(0, 0.05, 2).clip(-1, 1)
                # action = np.random.uniform(-1, 1, 2)
                obs, rewards, done, _, info = vec_env.step(action)
                returns += rewards
                steps += 1
        end = time.time()
        print(f'Episode {episode_no+1} finished after {steps} steps with return {returns:.2f}. Final x = {vec_env.sim.data.qpos[0]:.2f}')
        all_returns.append(returns)
        all_episode_lengths.append(steps)
        all_final_x.append(vec_env.sim.data.qpos[0])
        all_times.append((end - st) / steps)
        all_modes.append(modes)
    vec_env.close()
    print(f'Mean return: {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}')
    print(f'Mean episode length: {np.mean(all_episode_lengths):.2f} +/- {np.std(all_episode_lengths):.2f}')
    print(f'Mean final x: {np.mean(all_final_x):.2f} +/- {np.std(all_final_x):.2f}')
    print(f'Mean time per step: {np.mean(all_times):.2f} +/- {np.std(all_times):.2f}')
    return all_final_x, all_modes

def plot_x_histograms(final_x, labels=None):
    fig, ax = plt.subplots()
    track_end = 16.20
    bins = [0, track_end/4, track_end/2, 3*track_end/4, track_end, track_end+5]
    if len(final_x) > 1:
        ax.hist(final_x, bins=bins, label=labels)
    else:
        ax.hist(final_x, bins=bins)
    ax.legend()
    ax.set_xlabel('Final x position')
    ax.set_xticks(bins)
    ax.set_xticklabels(['0', '25%', '50%', '75%', '100%', 'Beyond'])
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of final x positions')
    plt.show()

def plot_modes(modes):
    fig, ax = plt.subplots()
    for i, mode in enumerate(modes):
        mode = np.array(mode)
        ax.scatter(mode[:, 0], mode[:, 1], label=f"Episode {i+1}")
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_title('Modes')
    ax.legend()
    plt.show()

def hyper_param_search(cfg: dict):
    cfg["steps"] = 122_880
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
                
                model.learn(total_timesteps=cfg["steps"], progress_bar=True, callback=eval_callback)
                
                model.save('sb3/hp/final')
                print("-------------------")

def train_sac(cfg: dict):
    env_fn = partial(make_env, cfg)
    vec_env = make_vec_env(env_fn, n_envs=12, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(env_fn, n_envs=1, vec_env_cls=SubprocVecEnv)
    sac_args = {}
    sac_args["learning_starts"] = 10_000
    sac_args["policy_kwargs"] = dict(activation_fn =nn.Tanh,
                    net_arch =[128, 128]
                    )
    eval_freq = 201
    model = SAC('MlpPolicy', vec_env, tensorboard_log=f'sb3/logs/{cfg["exp_name"]}', verbose=0, **sac_args)

    eval_callback = EvalCallback(eval_env, best_model_save_path=f'sb3/{cfg["exp_name"]}', log_path=f'sb3/{cfg["exp_name"]}', eval_freq=eval_freq, deterministic=True)
    
    model.learn(total_timesteps=cfg["steps"], progress_bar=True, callback=eval_callback)
    
    model.save(f'sb3/{cfg["exp_name"]}')

@hydra.main(version_base=None, config_name='dynamic', config_path=CONFIG_PATH)
def main(cfg: dict):
    cfg = OmegaConf.to_container(cfg)
    # train_ppo(cfg)
    ppo_final_x, ppo_modes = eval_model(cfg, algo="PPO", episodes=5)
    # plot_modes(ppo_modes)
    # sac_final_x = eval_model(cfg, algo="SAC")
    # final_xs = [ppo_final_x, sac_final_x]
    # plot_x_histograms(final_xs, labels=["PPO", "SAC"])
    # hyper_param_search(cfg)
    # train_sac(cfg)

if __name__ == '__main__':
    main()