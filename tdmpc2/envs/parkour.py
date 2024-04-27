import sys
sys.path.append('./')
from envs.parkour_base import ParkourEnv
import numpy as np
import torch
from common.parser import parse_cfg
import hydra
import nn as nn

PARKOUR_TASKS = ['parkour-base']

def make_env(cfg):
	"""
	Make Parkour environment.
	"""
	task = cfg.task
	if task not in PARKOUR_TASKS:
		raise ValueError('Unknown task:', task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	env = ParkourEnv(cfg, cfg.exp_conf_path)
	return env
