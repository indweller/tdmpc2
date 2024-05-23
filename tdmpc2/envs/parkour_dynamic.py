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

        for obstacle_key, obstacle_locs in self.obstacle_locs.items():
            for obstacle_loc in obstacle_locs:
                if obstacle_loc[0] <= base_pos[0] <= obstacle_loc[1]:
                    next_oh = self.get_next_obstacle_oh(obstacle_key, obstacle_loc)
                    high_level_obs = np.concatenate((high_level_obs, self.obstacle_oh[obstacle_key], next_oh))
                    return high_level_obs
        high_level_obs = np.concatenate((high_level_obs, np.array([-1, -1]), np.array([-1, -1])))
        return high_level_obs

    def get_next_obstacle_oh(self, current_obstacle_key, current_obstacle_loc):
        for obstacle_key, obstacle_locs in self.obstacle_locs.items():
            for obstacle_loc in obstacle_locs:
                if obstacle_loc != current_obstacle_loc and abs(obstacle_loc[0] - current_obstacle_loc[1] - self.gap) < 1e-3:
                    return self.obstacle_oh[obstacle_key]
        return np.array([-1, -1])

    def set_obstacles(self):
        obstacle_indices = [[24, 25, 26], [32, 33, 34], [43, 44, 45], [54, 55, 56]]
        obstacle_keys = ['rotating_disc', 'moving_cart', 'stewart_platform', 'stairs']
        obstacle_locs = {}

        x_coords = np.array([np.take(self.sim.data.qpos, indices)[0] for indices in obstacle_indices])
        x_coords += -0.5 * (np.arange(4) + 1)
        x_coords += self.gap * (np.arange(4) + 1)
        np.random.shuffle(x_coords)

        for i, (indices, key) in enumerate(zip(obstacle_indices, obstacle_keys)):
            obstacle = np.take(self.sim.data.qpos, indices)
            obstacle[0] = x_coords[i]
            np.put(self.sim.data.qpos, indices, obstacle)
            obstacle_locs[key] = [[x_coords[i] - 0.75, x_coords[i] + 0.75]]

        self.obstacle_locs = obstacle_locs
    
    def set_obstacles_new(self):
        obstacle_indices = [[24, 25, 26], [32, 33, 34], [43, 44, 45], [54, 55, 56]]
        obstacle_keys = ['rotating_disc', 'moving_cart', 'stewart_platform', 'stairs']
        obstacle_locs = {}

        obstacles = [np.take(self.sim.data.qpos, indices) for indices in obstacle_indices]

        x_coords = np.array([obstacles[i][0] for i in range(0, 3)])
        x_coords += -0.5 * (np.arange(3) + 1)
        x_coords += self.gap * (np.arange(3) + 1)
        np.random.shuffle(x_coords)
        for i in range(3):
            obstacles[i+1][0] = x_coords[i]

        obstacles[0][0] = -200

        for indices, obstacle, key in zip(obstacle_indices, obstacles, obstacle_keys):
            np.put(self.sim.data.qpos, indices, obstacle)
            obstacle_locs[key] = [[obstacle[0] - 0.75, obstacle[0] + 0.75]]

        self.obstacle_locs = obstacle_locs

    def set_obstacles_extended(self):
        obstacle_indices = [[32, 33, 34], [43, 44, 45], [54, 55, 56], 
                            [61, 62, 63], [72, 73, 74], [83, 84, 85], 
                            [90, 91, 92], [101, 102, 103], [112, 113, 114]]
        obstacle_keys = ['moving_cart', 'stewart_platform', 'stairs',
                         'moving_cart', 'stewart_platform', 'stairs',
                         'moving_cart', 'stewart_platform', 'stairs']
        obstacle_locs = {'rotating_disc': [[-200 - 0.75, -200 + 0.75]]}

        # Get current positions
        obstacles = [np.take(self.sim.data.qpos, indices) for indices in obstacle_indices]

        # Set rotating disc
        rotating_disc = np.take(self.sim.data.qpos, [24, 25, 26])
        rotating_disc[0] = -200
        np.put(self.sim.data.qpos, [24, 25, 26], rotating_disc)

        # Reset obstacles
        for i in range(9):
            obstacles[i][0] -= 2.0

        x_coords = np.array([obstacle[0] for obstacle in obstacles])
        x_coords += -0.5 * (np.arange(9) + 1)
        x_coords += self.gap * (np.arange(9) + 1)
        np.random.shuffle(x_coords)

        # Update positions in the simulation and in self.obstacle_locs
        for i, (indices, obstacle, key) in enumerate(zip(obstacle_indices, obstacles, obstacle_keys)):
            obstacle[0] = x_coords[i]
            np.put(self.sim.data.qpos, indices, obstacle)
            obstacle_locs[key] = obstacle_locs.get(key, []) + [[obstacle[0] - 0.75, obstacle[0] + 0.75]]

        self.obstacle_locs = obstacle_locs
        