from envs.parkour_base import ParkourEnv
import dtsd.envs.src.actions as act
import dtsd.envs.src.observations as obs
import numpy as np

class ParkourDynamic(ParkourEnv):
    def __init__(self, cfg, exp_conf_path='./exp_confs/default.yaml'):
        self.gap = 0.3
        self.obstacle_oh = {'rotating_disc': np.array([0, 0]), 'moving_cart': np.array([0, 1]), 'stewart_platform': np.array([1, 0]), 'stairs': np.array([1, 1])}
        self.obstacle_locs = {'rotating_disc': [[0, 0]], 'moving_cart': [[0, 0]], 'stewart_platform': [[0, 0]], 'stairs': [[0, 0]]}
        # self.obstacle_oh = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [-1, -1]])
        # self.obstacle_indices = np.array([[24, 25, 26], [32, 33, 34], [43, 44, 45], [54, 55, 56]])
        # self.obstacle_locs = np.zeros((4, 2))
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
            for _ in range(int(self.sim_freq / self.policy_freq)):
                if self.render_viewer:
                    self.sim.viewer.sync()
                torques = act.pd_targets(self, policy_action)
                torques[-1] = 0.05
                self.sim.set_control(torques)
                self.sim.simulate_n_steps(n_steps=1)
        current_pose = self.sim.data.qpos[:3].copy()
        reward = self.get_reward(prev_pose, current_pose)
        done = self.check_done(current_pose)
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
        return self.forward_reward_weight * np.exp(-0.15 * np.abs(current_pose[0] - 10))

    def generate_task_variant(self):
        self.set_obstacles_extended()
    
    def get_high_level_obs(self, mode_latent):
        high_level_obs = super().get_high_level_obs(mode_latent)
        base_pos = self.get_robot_base_pos()
        # for k in self.obstacle_locs.keys():
        #     if base_pos[0] >= self.obstacle_locs[k][0] and base_pos[0] <= self.obstacle_locs[k][1]:
        #         next_oh = np.array([-1, -1])
        #         for k2 in self.obstacle_locs.keys():
        #             if k2 != k and abs(self.obstacle_locs[k2][0] - self.obstacle_locs[k][1] - self.gap) < 1e-3:
        #                 next_oh = self.obstacle_oh[k2]
        #                 break
        #         high_level_obs = np.concatenate((high_level_obs, self.obstacle_oh[k], next_oh))
        #         return high_level_obs
        for k in self.obstacle_locs.keys():
            for k_i in self.obstacle_locs[k]:
                if base_pos[0] >= k_i[0] and base_pos[0] <= k_i[1]:
                    next_oh = np.array([-1, -1])
                    for k2 in self.obstacle_locs.keys():
                        for k2_i in self.obstacle_locs[k2]:
                            if k2_i != k_i and abs(k2_i[0] - k_i[1] - self.gap) < 1e-3:
                                next_oh = self.obstacle_oh[k2]
                                break
                    high_level_obs = np.concatenate((high_level_obs, self.obstacle_oh[k], next_oh))
                    return high_level_obs
        high_level_obs = np.concatenate((high_level_obs, np.array([-1, -1]), np.array([-1, -1])))
        return high_level_obs

    # def get_high_level_obs(self, mode_latent):
    #     high_level_obs = super().get_high_level_obs(mode_latent)
    #     base_pos = self.get_robot_base_pos()
    #     # Check if base_pos is within the location range of any obstacle
    #     in_range = np.logical_and(base_pos[0] >= self.obstacle_locs[:, 0], base_pos[0] <= self.obstacle_locs[:, 1])
    #     if np.any(in_range):
    #         # Find the indices of the current and next obstacles
    #         current_obstacle_index = np.argmax(in_range)
    #         # Concatenate the one-hot encodings of the current and next obstacles to the high-level observation
    #         high_level_obs = np.concatenate((high_level_obs, self.obstacle_oh[current_obstacle_index], self.obstacle_oh[current_obstacle_index + 1]))
    #     else:
    #         # Concatenate two [-1, -1] arrays to the high-level observation
    #         high_level_obs = np.concatenate((high_level_obs, np.array([-1, -1]), np.array([-1, -1])))
    #     return high_level_obs

    # def set_obstacles(self):
    #     # Get the x-coordinates of the obstacles
    #     x_coords = self.sim.data.qpos[self.obstacle_indices[:, 0]].copy()
    #     x_coords += -0.5 * np.arange(1, len(self.obstacle_indices) + 1)
    #     x_coords += self.gap * np.arange(1, len(self.obstacle_indices) + 1)
    #     # Shuffle the x-coordinates
    #     np.random.shuffle(x_coords)
    #     # Set the new x-coordinates of the obstacles
    #     self.sim.data.qpos[self.obstacle_indices[:, 0]] = x_coords
    #     # Update the obstacle locations
    #     self.obstacle_locs[:, 0] = x_coords - 0.75
    #     self.obstacle_locs[:, 1] = x_coords + 0.75  

    # def set_obstacles_new(self):
    #     # Get the x-coordinates of the obstacles
    #     x_coords = self.sim.data.qpos[self.obstacle_indices[:, 0]].copy()
    #     # Store the original first three locations
    #     original_first_three = x_coords[:3].copy()
    #     # Set the first obstacle to -200
    #     x_coords[0] = -200
    #     # Shuffle the original first three locations and assign them to the last three obstacles
    #     np.random.shuffle(original_first_three)
    #     x_coords[1:4] = original_first_three
    #     # Set the new positions
    #     self.sim.data.qpos[self.obstacle_indices[:, 0]] = x_coords
    #     # Update the obstacle locations
    #     self.obstacle_locs[:, 0] = x_coords - 0.75
    #     self.obstacle_locs[:, 1] = x_coords + 0.75

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
    
    def set_obstacles_new(self):
        rotating_disc =  np.take(self.sim.data.qpos, [24, 25, 26])
        moving_cart = np.take(self.sim.data.qpos, [32, 33, 34])
        stewart_platform = np.take(self.sim.data.qpos, [43, 44, 45])
        stairs = np.take(self.sim.data.qpos, [54, 55, 56])
        # rotating_disc, moving_cart, stewart_platform, stairs = obs.obstacle_state(self)
        stairs[0] = stewart_platform[0]
        stewart_platform[0] = moving_cart[0]
        moving_cart[0] = rotating_disc[0]
        rotating_disc[0] = 7.25
        x_coords = np.array([moving_cart[0], stewart_platform[0], stairs[0]])
        # x_coords -= 1.5
        x_coords += -0.5 * (np.arange(3) + 1)
        x_coords += self.gap * (np.arange(3) + 1)
        # Shuffle the x_coords
        np.random.shuffle(x_coords)
        # Set the new positions
        rotating_disc[0] = -200
        moving_cart[0] = x_coords[0]
        stewart_platform[0] = x_coords[1]
        stairs[0] = x_coords[2]
        # Set the new positions
        np.put(self.sim.data.qpos, [24, 25, 26], rotating_disc)
        np.put(self.sim.data.qpos, [32, 33, 34], moving_cart)
        np.put(self.sim.data.qpos, [43, 44, 45], stewart_platform)
        np.put(self.sim.data.qpos, [54, 55, 56], stairs)
        self.obstacle_locs = {'rotating_disc': [rotating_disc[0] - 0.75, rotating_disc[0] + 0.75], 'moving_cart': [moving_cart[0] - 0.75, moving_cart[0] + 0.75], 'stewart_platform': [stewart_platform[0] - 0.75, stewart_platform[0] + 0.75], 'stairs': [stairs[0] - 0.75, stairs[0] + 0.75]}

    def set_obstacles_extended(self):
        rotating_disc =  np.take(self.sim.data.qpos, [24, 25, 26])
        moving_cart = [np.take(self.sim.data.qpos, [32, 33, 34]), 
                       np.take(self.sim.data.qpos, [61, 62, 63]), 
                       np.take(self.sim.data.qpos, [90, 91, 92])]
        stewart_platform = [np.take(self.sim.data.qpos, [43, 44, 45]),
                            np.take(self.sim.data.qpos, [72, 73, 74]),
                            np.take(self.sim.data.qpos, [101, 102, 103])]
        stairs = [np.take(self.sim.data.qpos, [54, 55, 56]),
                  np.take(self.sim.data.qpos, [83, 84, 85]),
                  np.take(self.sim.data.qpos, [112, 113, 114])]
        
        rotating_disc[0] = -200
        np.put(self.sim.data.qpos, [24, 25, 26], rotating_disc)
        for i in range(3):
            moving_cart[i][0] -= 2.0
            stewart_platform[i][0] -= 2.0
            stairs[i][0] -= 2.0

        for i in range(3):
            x_coords = np.array([moving_cart[i][0], stewart_platform[i][0], stairs[i][0]])
            x_coords += -0.5 * (np.arange(i*3, (i+1)*3) + 1)
            x_coords += self.gap * (np.arange(i*3, (i+1)*3) + 1)
            np.random.shuffle(x_coords)
            moving_cart[i][0] = x_coords[0]
            stewart_platform[i][0] = x_coords[1]
            stairs[i][0] = x_coords[2]
            np.put(self.sim.data.qpos, [32 + 29 * i, 33 + 29 * i, 34 + 29 * i], moving_cart[i])
            np.put(self.sim.data.qpos, [43 + 29 * i, 44 + 29 * i, 45 + 29 * i], stewart_platform[i])
            np.put(self.sim.data.qpos, [54 + 29 * i, 55 + 29 * i, 56 + 29 * i], stairs[i])
        self.obstacle_locs = {'rotating_disc': [[rotating_disc[0] - 0.75, rotating_disc[0] + 0.75]], 'moving_cart': [[moving_cart[0][0] - 0.75, moving_cart[0][0] + 0.75], [moving_cart[1][0] - 0.75, moving_cart[1][0] + 0.75], [moving_cart[2][0] - 0.75, moving_cart[2][0] + 0.75]], 'stewart_platform': [[stewart_platform[0][0] - 0.75, stewart_platform[0][0] + 0.75], [stewart_platform[1][0] - 0.75, stewart_platform[1][0] + 0.75], [stewart_platform[2][0] - 0.75, stewart_platform[2][0] + 0.75]], 'stairs': [[stairs[0][0] - 0.75, stairs[0][0] + 0.75], [stairs[1][0] - 0.75, stairs[1][0] + 0.75], [stairs[2][0] - 0.75, stairs[2][0] + 0.75]]}
        