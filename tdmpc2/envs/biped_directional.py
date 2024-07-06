from envs.parkour_base import ParkourEnv
import dtsd.envs.src.actions as act
from dtsd.envs.src.env_logger import logger, logger_dummy
from dtsd.envs.src.env_frame_recorder import frame_recorder,frame_recorder_dummy
from dtsd.envs.src.misc_funcs import *
import numpy as np
import time
import datetime

class BipedDirectional(ParkourEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_episode_steps = 300
        self.episode_counter = 0
        self.curr_action = np.zeros(2)
        if cfg["high_level_reward"] == "pos_y":
            self.get_reward = self.pos_y_reward
        elif cfg["high_level_reward"] == "neg_x":
            self.get_reward = self.neg_x_reward

        this_exp_date = datetime.datetime.now().strftime("%d%b%Y")
        this_exp_time = datetime.datetime.now().strftime("%H:%M")
        if 'export_logger' in self.exp_conf.keys():
            self.exp_conf['export_logger']['export_date_time'] = this_exp_date + '/' + this_exp_time
            self.export_logger = logger(logger_conf=self.exp_conf['export_logger'])
        else:
            self.export_logger = logger_dummy(None)
        
        if exists_not_none('frame_recorder',self.exp_conf):
            self.record_frames = True
            self.sim.init_renderers()
            self.cam_trolly = { "pos" : np.array([4.05, 0, 0]),
                                "azim" : 90,
                                "elev" : -10,
                                "dist" : 2.25,
                                "delta_pos" : np.array([0.01,0.0,0.0]),
                                "delta_dist" : 0.105 }
            self.exp_conf['frame_recorder']['export_date_time'] = this_exp_date + '/' + this_exp_time
            self.frame_recorder = frame_recorder(self.exp_conf['frame_recorder'])
        else:
            self.record_frames = False
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
        if self.sim.data.qpos[2] < 0.3:
            return True
        if abs(self.sim.data.qpos[1]) > 0.75:
            return True
        if self.current_step >= self.max_episode_steps:
            return True
        return False

    def pos_y_reward(self, prev_pose, current_pose):
        return self.forward_reward_weight * np.exp(-0.08 * np.abs(current_pose[1] - 20))
    
    def neg_x_reward(self, prev_pose, current_pose):
        return self.forward_reward_weight * np.exp(-0.08 * np.abs(current_pose[0] + 20))

    def get_high_level_obs(self, mode_latent):
        low_level_obs = self.get_low_level_obs(mode_latent)
        return np.concatenate((low_level_obs, np.array([-1, -1]), np.array([-1, -1])))

    def update_rendering(self, start_time, end_time):
        time_to_sleep = max(0, 1 / self.policy_freq - (end_time - start_time))
        time.sleep(time_to_sleep)
        self.sim.viewer.sync()
        if self.record_frames:
            base_pos = self.get_robot_base_pos()
            self.cam_trolly["pos"][0] = base_pos[0]
            self.sim.update_camera( cam_name='free_camera',
                                    pos = self.cam_trolly["pos"],
                                    azim = self.cam_trolly["azim"],
                                    elev = self.cam_trolly["elev"],
                                    dist = self.cam_trolly["dist"],)	
            self.frame_recorder.append_frame(self.sim)
    
    def close(self):
        if self.render_viewer:
            self.sim.close()
        return super().close()