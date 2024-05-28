from dtsd.envs.src.transformations import euler_to_quat, quat_to_euler, quat_to_mat
from dtsd.envs.src.env_frame_recorder import frame_recorder, frame_recorder_dummy
from dtsd.envs.src.trajectory_class import biped_trajectory_preview
from dtsd.envs.src.env_logger import logger, logger_dummy
from dtsd.envs.sim.mujoco_sim_base import mujoco_sim
from dtsd.envs.src.metric2dist import metric2dist
from dtsd.envs.src.misc_funcs import *
from itertools import product
import dtsd.envs.src.observations as obs
import dtsd.envs.src.rewards as rew
import dtsd.envs.src.actions as act
import numpy as np
import importlib
import datetime
import torch
import yaml
import os
import gymnasium as gym
import sys
import nn as nn
sys.path.append('./')

NOMINAL_HEIGHT = 0.3


class ParkourEnv(gym.Env):

    def __init__(self, cfg):
        self.exp_conf = cfg
        self.render_viewer = self.exp_conf['sim_params']['render']
        self.policy = torch.load(self.exp_conf['policy_path'])
        self.oracle_freq = self.exp_conf['oracle_freq'] # Env freq in real time, 1 Hz
        self.policy_freq = self.exp_conf['policy_freq']
        self.sim = mujoco_sim( **self.exp_conf['sim_params'])
        # self.sim.init_renderers()
        if not isinstance(self.exp_conf['p_gain'],list):
            self.exp_conf['p_gain'] = [self.exp_conf['p_gain']]*10
        if not isinstance(self.exp_conf['d_gain'],list):
            self.exp_conf['d_gain'] = [self.exp_conf['d_gain']]*10
        self.phase = 0
        self.phaselen = int(self.policy_freq / self.oracle_freq)
        self.sim_freq = int(1 / self.sim.dt)
        model_prop_path = self.exp_conf['sim_params']['model_path'].replace('.xml','.yaml')
        prop_file = open(model_prop_path) 
        self.model_prop = yaml.load(prop_file, Loader=yaml.FullLoader)    
        self.action_space = gym.spaces.Box(
			low=np.array([-1, -1]), high=np.array([1, 1]),
		)
        x = self.exp_conf['d_ID'] / np.sqrt(2) - self.exp_conf['a_ID'][0]
        y = self.exp_conf['d_ID'] / np.sqrt(2) - self.exp_conf['a_ID'][1]
        self.x_lims = (self.exp_conf['a_ID'][0] - x, self.exp_conf['a_ID'][0] + x)
        self.y_lims = (self.exp_conf['a_ID'][1] - y, self.exp_conf['a_ID'][1] + y)
        self.action_space = gym.spaces.Box(
			low=np.array([-1, -1]), high=np.array([1, 1]),
		)
        self.xlen = self.exp_conf['observations']['terrain_xlen_infront']
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.get_high_level_obs(np.zeros(2)).shape, dtype=np.float64
        )
        self.forward_reward_weight = self.exp_conf['forward_reward_weight']
        self.max_episode_steps = 50
        self.current_step = 0

        self.transition_samplers = []
        for i in range(len(self.exp_conf['task']['modes'].keys())):
            self.transition_samplers.append(
                metric2dist(
                    n_cases=len(self.exp_conf['task']['modes'].keys()),
                    **self.exp_conf['task']
                )
            )

    def scale_actions(self, action):
        action[0] = (action[0] + 1) * (self.x_lims[1] - self.x_lims[0]) / 2 + self.x_lims[0]
        action[1] = (action[1] + 1) * (self.y_lims[1] - self.y_lims[0]) / 2 + self.y_lims[0]           
        return action

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
                self.sim.set_control(torques)
                self.sim.simulate_n_steps(n_steps=1)
        current_pose = self.sim.data.qpos[:3].copy()
        reward = self.get_reward(prev_pose, current_pose)
        done = False
        terrain_height = self.sim.get_terrain_height_at(current_pose)
        if self.sim.data.qpos[2] < 0.3 or self.sim.data.qpos[2] - terrain_height < 0.3:
            done = True
        if self.current_step >= self.max_episode_steps:
            done = True
        return self.get_high_level_obs(action), reward, done, {}
    
    def get_reward(self, prev_pose, current_pose):
        # reward = self.forward_reward_weight * (current_pose[2] - NOMINAL_HEIGHT)
        # reward = self.forward_reward_weight * (current_pose[0] - prev_pose[0]) / (self.phaselen * self.sim.dt * self.sim_freq / self.policy_freq)
        reward = self.forward_reward_weight * -1 * np.abs(current_pose[0] - 10)
        return reward

    def get_robot_base_pos(self):
        return self.sim.data.qpos[
                                self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][0]:
                                1+self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]          
                                ]

    def get_robot_base_tvel(self):
        return self.sim.data.qvel[
                                self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][0]:
                                1+self.model_prop[self.exp_conf['robot']]['ids']['base_tvel'][-1]          
                                ]

    def get_low_level_obs(self, mode_latent):
        robot_state = obs.robot_state(self)
        clock = obs.clock(self)
        return np.concatenate([robot_state, clock, mode_latent])
    
    def get_high_level_obs(self, mode_latent):
        low_level_obs = self.get_low_level_obs(mode_latent)
        terrain_scan = self.scan_terrain_xlen_infront(xlen=self.xlen).squeeze()
        return np.concatenate([low_level_obs, terrain_scan])

    def reset(self):
        self.current_step = 0
        self.sim.reset()
        self.constraint_robot()
        self.generate_task_variant()
        self.initialize_robot()
        if hasattr(self.policy, 'init_hidden_state'):
            self.policy.init_hidden_state()
        return self.get_high_level_obs(np.zeros(2))
    
    def constraint_robot(self): 
        # activate constraint, world_root constraint assumed to be 0
        self.sim.model.eq_active0[0] = 1    
        self.sim.data.eq_active[0] = 1

        # simulate to settle
        tvel_norm = np.inf
        while tvel_norm > 1e-3:
            self.sim.simulate_n_steps(1)
            base_tvel = self.get_robot_base_tvel()
            tvel_norm = np.linalg.norm(base_tvel)

        # set joint to nominal configuration
        jpn = np.array(self.model_prop[self.exp_conf['robot']]['jpos_nominal']) 
        error_norm = np.inf    
        while error_norm > 2e1:
            error_norm = 0
            for i,(jci,jpi,jvi) in enumerate(
                                                    zip(
                                                        self.model_prop[self.exp_conf['robot']]['ids']['jctrl'],                                                
                                                        self.model_prop[self.exp_conf['robot']]['ids']['jpos'],
                                                        self.model_prop[self.exp_conf['robot']]['ids']['jvel'],
                                                        )
                                                ):
                
                error_jpos = jpn[i] - self.sim.data.qpos[jpi]
                error_jvel = 0 - self.sim.data.qvel[jvi]
                error_norm += error_jpos**2 + error_jvel**2
                self.sim.data.ctrl[jci] =  self.exp_conf['p_gain'][i]*(error_jpos) \
                            + self.exp_conf['d_gain'][i]*(error_jvel)
            error_norm = np.sqrt(error_norm)
            self.sim.simulate_n_steps(1)
      
    def initialize_robot(self):

        # deactivate constraint, world_root constraint assumed to be 0
        self.sim.model.eq_active0[0] = 0      
        self.sim.data.eq_active[0] = 0

        # zero our velocity as the ideal initial condition
        self.sim.data.qvel[:] = 0.0    

        # base state initialisation
        # set robot to nominal height 
        self.sim.data.qpos[self.model_prop[self.exp_conf['robot']]['ids']['base_pos'][-1]] = self.model_prop[self.exp_conf['robot']]['height_nominal']

        # one step to settle
        self.sim.simulate_n_steps(1)
    
    def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
        return np.zeros((width, height, 3), dtype=np.uint8)
        # return self.sim.render(mode, width, height, camera_id)
        # return self.sim.get_frame_from_renderer()

    def generate_task_variant(self):

        if exists_not_none('track_x_start', self.exp_conf['task']):
            track_xlen = self.exp_conf['task']['track_x_start']
        else:
            track_xlen = 0.0

        # choose the starting mode
        curr_mode_id = np.random.randint(
            low=0,
            high=len(self.exp_conf['task']['modes'].keys())
        )
        while track_xlen < self.exp_conf['task']['track_x_length']:
            # get the distribution of the current mode
            te_prob_dist = self.transition_samplers[curr_mode_id].return_prob()
            # sample the nex mode
            curr_mode_name = np.random.choice(
                list(self.exp_conf['task']['modes'].keys()),
                p=te_prob_dist
            )
            # update mode id
            curr_mode_id = list(
                self.exp_conf['task']['modes'].keys()).index(curr_mode_name)

            # chose the mode variant
            curr_mode_dict = self.exp_conf['task']['modes'][curr_mode_name]
            # sampl the mode paramters
            if 'discrete' in curr_mode_dict['param_dist']['type']:
                curr_mode_param = np.random.choice(
                    curr_mode_dict['param_dist']['points'])
            elif curr_mode_dict['param_dist']['type'] == 'continuous':
                curr_mode_param = np.random.uniform(
                    low=curr_mode_dict['param_dist']['support'][0],
                    high=curr_mode_dict['param_dist']['support'][-1]
                )

            if exists_and_true('manipulate_terrain', curr_mode_dict):
                # for gap and blocks
                gb_start = track_xlen+curr_mode_param[0]
                gb_end = track_xlen+curr_mode_param[0]+curr_mode_param[1]
                gb_height = curr_mode_param[2]

                self.sim.generate_terrain_plateau(
                    gb_start+0.05,
                    gb_end+0.05,
                    -0.5, 0.5,
                    gb_height
                )
                track_xlen += curr_mode_param[0]+curr_mode_param[1]
            else:
                # for flat
                track_xlen += curr_mode_param[0]  # goal_x
    
    def scan_terrain_xlen_infront(self,xlen=1.0,return_start_pos=False):

        base_pos = self.get_robot_base_pos()
        terrain_map = self.sim.get_terrain_infront(
                                                    pos=base_pos,
                                                    halfwidth_x=xlen,
                                                    halfwidth_y=0.08
                                                    )
        x_half_index = int(0.5*terrain_map.shape[0])
        terrain_map = terrain_map[x_half_index:,0:1]

        if return_start_pos:
            return terrain_map,base_pos

        return terrain_map
