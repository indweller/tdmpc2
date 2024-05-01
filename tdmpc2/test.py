import numpy as np
import torch
import hydra
from common.parser import parse_cfg
from envs import make_env

@hydra.main(version_base=None, config_name='config_dynamic', config_path='/home/learning/prashanth/tdmpc2/tdmpc2/')
def main(cfg: dict):
    import yaml
    import sys
    sys.path.append('./')
    # cfg = yaml.load(open('/home/learning/prashanth/tdmpc2/tdmpc2/config.yaml'), Loader=yaml.FullLoader)
    cfg = parse_cfg(cfg)
    env = make_env(cfg)
    exp_conf = yaml.load(open(cfg.exp_conf_path), Loader=yaml.FullLoader)
    render = exp_conf['sim_params']['render']
    all_returns = []
    all_episode_lengths = []
    for i in range(20):
        obs	= env.reset()
        env.sim.viewer_paused = render # pause the viewer
        if render:
            env.sim.viewer.update_hfield(0) # update the heightfield
        # episode counters
        done = False
        steps = 0
        returns = 0
        while not done:
            # simulate if not paused
            if not env.sim.viewer_paused:
            # set cameras and sync with viewer
                base_pos = env.get_robot_base_pos()
                # action = np.random.uniform(-1, 1, 2)
                # Gaussian around the center``
                action = np.random.normal(0, 0.05, 2).clip(-1, 1)
                action = torch.tensor(action)
                next_obs, reward, done, info_dict = env.step(action)
                obs = torch.Tensor(next_obs)
                steps += 1
                returns += reward
                if done:
                    print(f'Episode finished after {steps} steps with return {returns}')
                    # time.sleep(2)
        all_returns.append(returns)
        all_episode_lengths.append(steps)
    print(f'Mean return: {np.mean(all_returns)} +/- {np.std(all_returns)}')
    print(f'Mean episode length: {np.mean(all_episode_lengths)} +/- {np.std(all_episode_lengths)}')

if __name__ == '__main__':
    main()