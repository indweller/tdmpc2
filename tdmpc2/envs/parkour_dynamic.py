from envs.parkour_base import ParkourEnv
import dtsd.envs.src.actions as act
import dtsd.envs.src.observations as obs
import numpy as np

class ParkourDynamic(ParkourEnv):
    def __init__(self, cfg, exp_conf_path='./exp_confs/default.yaml'):
        self.gap = 0.3
        self.obstacle_oh = {'rotating_disc': np.array([0, 0]), 'moving_cart': np.array([0, 1]), 'stewart_platform': np.array([1, 0]), 'stairs': np.array([1, 1])}
        self.obstacle_locs = {'rotating_disc': [0, 0], 'moving_cart': [0, 0], 'stewart_platform': [0, 0], 'stairs': [0, 0]}
        super().__init__(cfg, exp_conf_path)
        self.max_episode_steps = 10
    
    def step(self, action):
        self.current_step += 1
        action = self.scale_actions(action)
        prev_pose = self.sim.data.qpos[:3].copy()
        for phase in range(self.phaselen):
            self.phase = phase
            low_level_obs = self.get_low_level_obs(action)
            policy_action = self.policy(low_level_obs).detach().numpy()
            for _ in range(int(self.sim_freq / self.policy_freq)):
                if self.render_viewer:
                    self.sim.viewer.sync()
                torques = act.pd_targets(self, policy_action)
                torques[-1] = 0.05
                self.sim.set_control(torques)
                self.sim.simulate_n_steps(n_steps=1)
        current_pose = self.sim.data.qpos[:3].copy()
        reward = self.get_reward(prev_pose, current_pose)
        done = False
        terrain_height = self.sim.get_terrain_height_at(current_pose)
        if self.sim.data.qpos[2] < 0.35 or self.sim.data.qpos[2] - terrain_height < 0.35:
            done = True
        if abs(self.sim.data.qpos[1]) > 0.75:
            done = True
        if self.current_step >= self.max_episode_steps:
            done = True
        return self.get_high_level_obs(action), reward, done, {}

    def get_reward(self, prev_pose, current_pose):
        return self.forward_reward_weight * np.exp(-0.15 * np.abs(current_pose[0] - 10))

    def generate_task_variant(self):
        self.set_obstacles()
    
    def get_high_level_obs(self, mode_latent):
        high_level_obs = super().get_high_level_obs(mode_latent)
        base_pos = self.get_robot_base_pos()
        for k in self.obstacle_locs.keys():
            if base_pos[0] >= self.obstacle_locs[k][0] and base_pos[0] <= self.obstacle_locs[k][1]:
                next_oh = np.array([-1, -1])
                for k2 in self.obstacle_locs.keys():
                    if k2 != k and abs(self.obstacle_locs[k2][0] - self.obstacle_locs[k][1] - self.gap) < 1e-3:
                        next_oh = self.obstacle_oh[k2]
                        break
                high_level_obs = np.concatenate((high_level_obs, self.obstacle_oh[k], next_oh))
                return high_level_obs
        high_level_obs = np.concatenate((high_level_obs, np.array([-1, -1]), np.array([-1, -1])))
        return high_level_obs

    def set_obstacles(self):
        rotating_disc =  np.take(self.sim.data.qpos, [24, 25, 26])
        moving_cart = np.take(self.sim.data.qpos, [32, 33, 34])
        stewart_platform = np.take(self.sim.data.qpos, [43, 44, 45])
        stairs = np.take(self.sim.data.qpos, [54, 55, 56])
        # rotating_disc, moving_cart, stewart_platform, stairs = obs.obstacle_state(self)
        x_coords = np.array([rotating_disc[0], moving_cart[0], stewart_platform[0], stairs[0]])
        x_coords += -0.5 * (np.arange(4) + 1)
        x_coords += self.gap * (np.arange(4) + 1)
        # Shuffle the x_coords
        np.random.shuffle(x_coords)
        # Set the new positions
        rotating_disc[0] = x_coords[0]
        moving_cart[0] = x_coords[1]
        stewart_platform[0] = x_coords[2]
        stairs[0] = x_coords[3]
        # Set the new positions
        np.put(self.sim.data.qpos, [24, 25, 26], rotating_disc)
        np.put(self.sim.data.qpos, [32, 33, 34], moving_cart)
        np.put(self.sim.data.qpos, [43, 44, 45], stewart_platform)
        np.put(self.sim.data.qpos, [54, 55, 56], stairs)
        self.obstacle_locs = {'rotating_disc': [rotating_disc[0] - 0.75, rotating_disc[0] + 0.75], 'moving_cart': [moving_cart[0] - 0.75, moving_cart[0] + 0.75], 'stewart_platform': [stewart_platform[0] - 0.75, stewart_platform[0] + 0.75], 'stairs': [stairs[0] - 0.75, stairs[0] + 0.75]}
        


