from envs.parkour_dynamic import ParkourDynamic
import dtsd.envs.src.observations as obs
import dtsd.envs.src.actions as act
import numpy as np
import gymnasium as gym
import time

class ParkourDynamicEnd2End(ParkourDynamic):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.action_space = gym.spaces.Box(
			low=np.array([-30, -30, -30, -50, -30,
                         -30, -30, -30, -50, -30,]), 
            high=np.array([30, 30, 30, 50, 30,
                         30, 30, 30, 50, 30,]),
		)
        self.max_episode_steps = 900

    def step(self, action):
        self.current_step += 1
        prev_pose = self.sim.data.qpos[:3].copy()
        start_time = time.time()
        for i in range(int(self.sim_freq / self.policy_freq)):
            torques = act.pd_targets(self, action)
            torques[-1] = 0.0
            self.sim.set_control(torques)
            self.sim.simulate_n_steps(n_steps=1)
            self.export_logger.update(self, self.sim.data)
            done = self.check_done(self.sim.data.qpos[:3])
            if done:
                break
        end_time = time.time()
        if self.render_viewer:
            self.update_rendering(start_time, end_time)            
        if done:
            export_name = "epi_" + str(self.episode_counter)
            self.frame_recorder.export(export_name = export_name)
            self.export_logger.export(export_name = export_name)
        current_pose = self.sim.data.qpos[:3].copy()
        reward = self.get_reward(prev_pose, current_pose)
        return self.get_high_level_obs(action), reward, done, {}
        
    def get_high_level_obs(self, action=None):
        robot_state = obs.robot_state(self)
        clock = obs.clock(self)
        terrain_scan = self.scan_terrain_xlen_infront(xlen=self.xlen).squeeze()
        low_level_obs = np.concatenate([robot_state, clock, terrain_scan])
        if self.obstacle_locs is None:
            return np.concatenate((low_level_obs, np.array([-1, -1]), np.array([-1, -1])))
        base_pos = self.get_robot_base_pos()
        x = base_pos[0]
        oh = np.array([-1, -1])
        current_obstacle = self.get_current_obstacle()
        if current_obstacle is not None:
            oh = self.obstacle_oh[current_obstacle]
        next_oh = np.array([-1, -1])
        next_obstacle = self.get_next_obstacle()
        if next_obstacle is not None:
            next_oh = self.obstacle_oh[next_obstacle]
        return np.concatenate((low_level_obs, oh, next_oh))
