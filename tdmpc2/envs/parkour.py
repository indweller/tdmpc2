import sys
sys.path.append('./')
from envs.parkour_base import ParkourEnv
from envs.parkour_dynamic import ParkourDynamic
import nn as nn

PARKOUR_TASKS = ['parkour-base', 'parkour-dynamic']

def make_env(cfg):
	"""
	Make Parkour environment.
	"""
	task = cfg.task
	if task not in PARKOUR_TASKS:
		raise ValueError('Unknown task:', task)
	env_cls = {
		'parkour-base': ParkourEnv,
		'parkour-dynamic': ParkourDynamic,
	}[task]
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
	# env = ParkourEnv(cfg, cfg.exp_conf_path)
	env = env_cls(cfg, cfg.exp_conf_path)
	return env
