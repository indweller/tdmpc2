import torch.nn as nn
import numpy as np
from envs.parkour import ParkourEnv
from envs.parkour_dynamic import ParkourDynamic
from envs.biped_directional import BipedDirectional
from envs.parkour_dynamic_end2end import ParkourDynamicEnd2End
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
        self.max_episode_steps = 300
        
    def step(self, action):
        obs, reward, done, info = super(PPOParkourDynamicEnv, self).step(action)
        return obs, reward, done, False, info
    
    def reset(self, seed=0):
        return super(PPOParkourDynamicEnv, self).reset(), {}

class PPOBipedDirectionalEnv(BipedDirectional):
    def __init__(self, cfg):
        super(PPOBipedDirectionalEnv, self).__init__(cfg)
    
    def step(self, action):
        obs, reward, done, info = super(PPOBipedDirectionalEnv, self).step(action)
        return obs, reward, done, False, info
    
    def reset(self, seed=0):
        return super(PPOBipedDirectionalEnv, self).reset(), {}

class PPOParkourDynamicEnd2EndEnv(ParkourDynamicEnd2End):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, False, info
    
    def reset(self, seed=0):
        return super().reset(), {}
    
def make_env(cfg):
    env = PPOParkourDynamicEnd2EndEnv(cfg)
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

def eval_model(cfg, algo="PPO", version=1, episodes=50):
    env_fn = partial(make_env, cfg)
    vec_env = env_fn()
    if algo == "PPO":
        print("Loading PPO model")
        # model = PPO.load('sb3/hp/50_64_32/best_model.zip')
        model = PPO.load(f'../../discovery/sb3/scaling/ppo_{version}x_high/best_model.zip')
    else:
        print("Loading SAC model")
        model = SAC.load(f'../../discovery/sb3/scaling/sac_{version}x_high/best_model.zip')
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
    all_actions = []
    for episode_no in range(episodes):
        obs, _ = vec_env.reset()
        done = False
        if render:
            # vec_env.sim.viewer_paused = True
            vec_env.sim.viewer.update_hfield(0)
        returns = 0
        steps = 0
        modes = []
        actions = []
        import time
        st = time.time()
        while not done:
            if render and not vec_env.sim.viewer_paused:
                # print(model.predict(obs))
                action, _states = model.predict(obs, deterministic=True)
                modes.append(vec_env.scale_actions(action))
                actions.append(action)
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
        all_actions.append(actions)
    vec_env.close()
    print(f'Mean return: {np.mean(all_returns):.2f} +/- {np.std(all_returns):.2f}')
    print(f'Mean episode length: {np.mean(all_episode_lengths):.2f} +/- {np.std(all_episode_lengths):.2f}')
    print(f'Mean final x: {np.mean(all_final_x):.2f} +/- {np.std(all_final_x):.2f}')
    print(f'Mean time per step: {np.mean(all_times):.2f} +/- {np.std(all_times):.2f}')
    return all_final_x, all_modes, all_actions

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

def plot_modes(cfg, modes, actions, labels=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    for i, mode in enumerate(modes):
        mode = np.array(np.concatenate(mode))
        action = np.array(np.concatenate(actions[i]))
        print(mode.shape, action.shape)
        ax.scatter(mode[:, 0], mode[:, 1], label=labels[i], marker='o')
        ax.scatter(action[:, 0], action[:, 1], label=labels[i], marker='x')
    a_ID = cfg["a_ID"]
    d_ID = cfg["d_ID"]
    ax.add_patch(plt.Circle(a_ID, d_ID, fill=False, color='black'))
    # Draw a boundary along (-1, -1), (-1, 1), (1, 1), (1, -1)
    ax.plot([-1, -1], [-1, 1], color='black')
    ax.plot([-1, 1], [1, 1], color='black')
    ax.plot([1, 1], [1, -1], color='black')
    ax.plot([1, -1], [-1, -1], color='black')
    # Draw a boundary with center at a_ID and width d_ID/sqrt(2)
    ax.plot([a_ID[0] - d_ID/np.sqrt(2), a_ID[0] - d_ID/np.sqrt(2)], [a_ID[1] - d_ID/np.sqrt(2), a_ID[1] + d_ID/np.sqrt(2)], color='red')
    ax.plot([a_ID[0] - d_ID/np.sqrt(2), a_ID[0] + d_ID/np.sqrt(2)], [a_ID[1] + d_ID/np.sqrt(2), a_ID[1] + d_ID/np.sqrt(2)], color='red')
    ax.plot([a_ID[0] + d_ID/np.sqrt(2), a_ID[0] + d_ID/np.sqrt(2)], [a_ID[1] + d_ID/np.sqrt(2), a_ID[1] - d_ID/np.sqrt(2)], color='red')
    ax.plot([a_ID[0] + d_ID/np.sqrt(2), a_ID[0] - d_ID/np.sqrt(2)], [a_ID[1] - d_ID/np.sqrt(2), a_ID[1] - d_ID/np.sqrt(2)], color='red')    
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_title('Modes')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
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
    print(cfg["exp_name"])
    # ppo_1x, ppo_1x_modes, ppo_1x_actions = eval_model(cfg, algo="PPO", version=1, episodes=50)
    # sac_1x, sac_1x_modes, sac_1x_actions = eval_model(cfg, algo="SAC", version=1, episodes=50)
    cfg["d_ID"] = 2 * cfg["d_ID"]
    ppo_2x, ppo_2x_modes, ppo_2x_actions = eval_model(cfg, algo="PPO", version=2, episodes=50)
    # sac_5x, sac_5x_modes, sac_5x_actions = eval_model(cfg, algo="SAC", version=5, episodes=50)
    # final_xs = [ppo_1x, ppo_5x, sac_1x, sac_5x]
    # final_modes = [ppo_1x_modes, ppo_5x_modes, sac_1x_modes, sac_5x_modes]
    # final_actions = [ppo_1x_actions, ppo_5x_actions, sac_1x_actions, sac_5x_actions]
    # np.save(f'sb3/scaling_final_xs.npy', final_xs)
    # np.save(f'sb3/scaling_final_modes.npy', final_modes)
    # np.save(f'sb3/scaling_final_actions.npy', final_actions)
    # plot_x_histograms(final_xs, labels=["PPO 1x", "PPO 5x", "SAC 1x", "SAC 5x"])
    # plot_modes(cfg, final_modes, final_actions)
    # hyper_param_search(cfg)
    # train_ppo(cfg)
    # train_sac(cfg)

if __name__ == '__main__':
    main()