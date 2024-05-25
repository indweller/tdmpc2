from envs.parkour_base import ParkourEnv
import dtsd.envs.src.actions as act
import dtsd.envs.src.observations as obs
import numpy as np
import time

class ParkourDynamic(ParkourEnv):
    def __init__(self, cfg, exp_conf_path='./exp_confs/default.yaml'):
        self.gap = 0.3
        self.obstacle_oh = {'rotating_disc': np.array([0, 0]), 'moving_cart': np.array([0, 1]), 'stewart_platform': np.array([1, 0]), 'stairs': np.array([1, 1])}
        self.obstacle_locs = None
        self.obstacle_keys = ['moving_cart', 'stewart_platform', 'stairs', 'moving_cart', 'stewart_platform', 'stairs', 'moving_cart', 'stewart_platform', 'stairs']
        super().__init__(cfg, exp_conf_path)
        self.max_episode_steps = 30
    
    def step(self, action):
        self.current_step += 1
        action = self.scale_actions(action)
        prev_pose = self.sim.data.qpos[:3].copy()
        for phase in range(self.phaselen):
            self.phase = phase
            low_level_obs = self.get_low_level_obs(action)
            policy_action = self.policy(low_level_obs).detach().numpy()
            st = time.time()
            for i in range(int(self.sim_freq / self.policy_freq)):
                torques = act.pd_targets(self, policy_action)
                torques[-1] = 0.00
                self.sim.set_control(torques)
                self.sim.simulate_n_steps(n_steps=1)
                done = self.check_done(self.sim.data.qpos[:3])
                if done:
                    break
            end = time.time()
            if self.render_viewer:
                time_to_sleep = max(0, 1 / self.policy_freq - (end - st))
                time.sleep(time_to_sleep)
                self.sim.viewer.sync()
        current_pose = self.sim.data.qpos[:3].copy()
        reward = self.get_reward(prev_pose, current_pose)
        return self.get_high_level_obs(action), reward, done, {}
    
    def check_done(self, current_pose):
        terrain_height = self.sim.get_terrain_height_at(current_pose)
        if self.sim.data.qpos[2] < 0.35 or self.sim.data.qpos[2] - terrain_height < 0.35:
            return True
        if abs(self.sim.data.qpos[1]) > 0.75:
            return True
        if self.current_step >= self.max_episode_steps:
            return True
        return False

    def get_reward(self, prev_pose, current_pose):
        # return self.forward_reward_weight * np.exp(-0.15 * np.abs(current_pose[0] - 10))
        return self.forward_reward_weight * np.exp(-0.08 * np.abs(current_pose[0] - 20))

    def generate_task_variant(self):
        self.set_obstacles_extended()
    
    def get_high_level_obs(self, mode_latent):
        high_level_obs = super().get_high_level_obs(mode_latent)
        if self.obstacle_locs is None:
            return np.concatenate((high_level_obs, np.array([-1, -1]), np.array([-1, -1])))
        base_pos = self.get_robot_base_pos()
        x = base_pos[0]
        oh = np.array([-1, -1])
        for i, loc in enumerate(self.obstacle_locs):
            if loc - 0.75 <= x <= loc + 0.75:
                oh = self.obstacle_oh[self.obstacle_keys[i]]
                break
        next_oh = np.array([-1, -1])
        for i, loc in enumerate(self.obstacle_locs):
            if loc - 0.75 > x:
                next_oh = self.obstacle_oh[self.obstacle_keys[i]]
                break
        return np.concatenate((high_level_obs, oh, next_oh))
    
    def set_obstacles(self):
        obstacle_indices = [[24, 25, 26], [32, 33, 34], [43, 44, 45], [54, 55, 56]]
        x_coords = np.array([np.take(self.sim.data.qpos, indices)[0] for indices in obstacle_indices])
        obstacle_keys = ['rotating_disc', 'moving_cart', 'stewart_platform', 'stairs']
        obstacle_indices = {'rotating_disc': [24, 25, 26],
                            'moving_cart': [32, 33, 34],
                            'stewart_platform': [43, 44, 45],
                            'stairs': [54, 55, 56]}

        x_coords += -0.5 * (np.arange(4) + 1)
        x_coords += self.gap * (np.arange(4) + 1)
        np.random.shuffle(obstacle_keys)

        for i, key in enumerate(obstacle_keys):
            index = obstacle_indices[key]
            obstacles = np.take(self.sim.data.qpos, index)
            obstacles[0] = x_coords[i]
            np.put(self.sim.data.qpos, index, obstacles)
        
        self.obstacle_locs = x_coords
        self.obstacle_keys = obstacle_keys
    
    def set_obstacles_new(self):
        obstacle_indices = [[24, 25, 26], [32, 33, 34], [43, 44, 45], [54, 55, 56]]
        x_coords = np.array([np.take(self.sim.data.qpos, indices)[0] for indices in obstacle_indices])
        obstacle_indices = {'moving_cart': [32, 33, 34],
                            'stewart_platform': [43, 44, 45],
                            'stairs': [54, 55, 56]}
        obstacle_keys = ['moving_cart', 'stewart_platform', 'stairs']

        x_coords = x_coords[:3]
        x_coords += -0.5 * (np.arange(3) + 1)
        x_coords += self.gap * (np.arange(3) + 1)
        np.random.shuffle(obstacle_keys)

        for i, key in enumerate(obstacle_keys):
            index = obstacle_indices[key]
            obstacles = np.take(self.sim.data.qpos, index)
            obstacles[0] = x_coords[i]
            np.put(self.sim.data.qpos, index, obstacles)
        np.put(self.sim.data.qpos, [24], [20])
        
        self.obstacle_locs = x_coords
        self.obstacle_keys = obstacle_keys

    def set_obstacles_extended(self):
        obstacle_indices = [[32, 33, 34], [43, 44, 45], [54, 55, 56], 
                            [61, 62, 63], [72, 73, 74], [83, 84, 85], 
                            [90, 91, 92], [101, 102, 103], [112, 113, 114]]
        x_coords = np.array([np.take(self.sim.data.qpos, indices)[0] for indices in obstacle_indices])
        obstacle_indices = {'moving_cart': [[32, 33, 34], [61, 62, 63], [90, 91, 92]],
                            'stewart_platform': [[43, 44, 45], [72, 73, 74], [101, 102, 103]],
                            'stairs': [[54, 55, 56], [83, 84, 85], [112, 113, 114]]}
        obstacle_keys = ['moving_cart', 'stewart_platform', 'stairs']*3

        x_coords -= 2.0
        x_coords += -0.5 * (np.arange(9) + 1)
        x_coords += self.gap * (np.arange(9) + 1)
        np.random.shuffle(obstacle_keys)

        for i, key in enumerate(obstacle_keys):
            index = obstacle_indices[key].pop()
            obstacles = np.take(self.sim.data.qpos, index)
            obstacles[0] = x_coords[i]
            np.put(self.sim.data.qpos, index, obstacles)
        np.put(self.sim.data.qpos, [24], [2000])
        
        self.obstacle_locs = x_coords
        self.obstacle_keys = obstacle_keys