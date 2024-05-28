from envs.parkour_base import ParkourEnv
import dtsd.envs.src.actions as act
from dtsd.envs.src.env_logger import logger, logger_dummy
from dtsd.envs.src.env_frame_recorder import frame_recorder,frame_recorder_dummy
from dtsd.envs.src.misc_funcs import *
import numpy as np
import time
import datetime

class ParkourDynamic(ParkourEnv):
    def __init__(self, cfg):
        self.gap = 0.3
        self.obstacle_oh = {'rotating_disc': np.array([0, 0]), 'moving_cart': np.array([0, 1]), 'stewart_platform': np.array([1, 0]), 'stairs': np.array([1, 1])}
        self.obstacle_locs = None
        self.obstacle_keys = ['moving_cart', 'stewart_platform', 'stairs', 'moving_cart', 'stewart_platform', 'stairs', 'moving_cart', 'stewart_platform', 'stairs']
        
        super().__init__(cfg)
        
        self.max_episode_steps = 30
        self.episode_counter = 0
        self.curr_action = np.zeros(2)

        if 'export_logger' in self.exp_conf.keys():
            this_exp_date = datetime.datetime.now().strftime("%d%b%Y")
            this_exp_time = datetime.datetime.now().strftime("%H:%M")
            self.exp_conf['export_logger']['export_date_time'] = this_exp_date + '/' + this_exp_time
            self.export_logger = logger(logger_conf=self.exp_conf['export_logger'])
        else:
            self.export_logger = logger_dummy(None)
        
        if exists_not_none('frame_recorder',self.exp_conf):
            self.sim.init_renderers()
            self.cam_trolly = { "pos" : np.array([4.05, 0, 0]),
                                "azim" : 90,
                                "elev" : -10,
                                "dist" : 6.5,
                                "delta_pos" : np.array([0.01,0.0,0.0]),
                                "delta_dist" : 0.105 }
            self.exp_conf['frame_recorder']['export_date_time'] = this_exp_date + '/' + this_exp_time
            self.frame_recorder = frame_recorder(self.exp_conf['frame_recorder'])
        else:
            self.frame_recorder = frame_recorder_dummy(None)
    
    def reset(self):
        self.export_logger.reset()
        self.frame_recorder.reset()
        self.episode_counter += 1
        return super().reset()
    
    def step(self, action):
        self.curr_action = action.copy()
        self.current_step += 1
        action = self.scale_actions(action)
        prev_pose = self.sim.data.qpos[:3].copy()
        for phase in range(self.phaselen):
            self.phase = phase
            low_level_obs = self.get_low_level_obs(action)
            policy_action = self.policy(low_level_obs).detach().numpy()
            start_time = time.time()
            for i in range(int(self.sim_freq / self.policy_freq)):
                torques = act.pd_targets(self, policy_action)
                torques[-1] = 0.00
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
                break
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

    def update_rendering(self, start_time, end_time):
        time_to_sleep = max(0, 1 / self.policy_freq - (end_time - start_time))
        time.sleep(time_to_sleep)
        self.sim.viewer.sync()
        base_pos = self.get_robot_base_pos()
        self.cam_trolly["pos"][0] = base_pos[0]
        self.sim.update_camera( cam_name='free_camera',
                                pos = self.cam_trolly["pos"],
                                azim = self.cam_trolly["azim"],
                                elev = self.cam_trolly["elev"],
                                dist = self.cam_trolly["dist"],)	
        self.frame_recorder.append_frame(self.sim)
        
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