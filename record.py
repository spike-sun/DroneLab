##########################
#    Launch Isaac Sim    #
##########################

import os
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record IL dataset.")
parser.add_argument("--task", type=str, default="ForestChaser")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--use_fabric", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ['ENABLE_CAMERAS'] = str(int(args_cli.enable_cameras))
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#######################
#    Anything else    #
#######################

import h5py
from tqdm import tqdm
import torch
import numpy as np
from omegaconf import OmegaConf
import gymnasium as gym
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from skrl.envs.wrappers.torch import IsaacLabWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
from models.teacher import TeacherPolicy
import skrl.utils
from skrl.utils.spaces.torch import unflatten_tensorized_space
from skrl.resources.preprocessors.torch import RunningStandardScaler
import envs




def append_dict_to_hdf5(h5_group: h5py.Group, data_dict: dict):
    for k, v in data_dict.items():
        if isinstance(v, dict):
            subgroup = h5_group.require_group(k)  # will create group if not exist
            append_dict_to_hdf5(subgroup, v)
        elif isinstance(v, torch.Tensor):
            if k in h5_group:
                # append to dataset
                data = v.cpu().numpy()  # (B, *S)
                dataset = h5_group[k]
                assert dataset.shape[1:] == data.shape
                cur_size = dataset.shape[0]
                dataset.resize((cur_size+1, *dataset.shape[1:]))  # (T, B, *S)
                dataset[-1] = data
            else:
                # create if dataset not exist
                data = v.unsqueeze(0).cpu().numpy()  # (1, B, *S)
                dataset = h5_group.create_dataset(
                    name=k,
                    data=data,
                    shape=data.shape,
                    maxshape=(None, *data.shape[1:])
                )
        else:
            raise TypeError(f"Unsupported data type for key '{k}': {type(v)}")


log_dir = "data/dagger"
device = args_cli.device
num_envs = args_cli.num_envs
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=args_cli.use_fabric)
env_cfg.seed = args_cli.seed
skrl.utils.set_seed(args_cli.seed)
dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)

env = gym.make(args_cli.task, cfg=env_cfg)
env = IsaacLabWrapper(env)

weights = torch.load('logs/policy/teacher/PPO_2025-02-14_00-18-22/checkpoints/best_agent.pt', weights_only=True)
teacher_state_preprocessor = RunningStandardScaler(size=env.observation_space, device=device).to(device)
teacher_state_preprocessor.load_state_dict(weights['state_preprocessor'])
teacher_state_preprocessor.eval()
teacher_policy = TeacherPolicy(observation_space=env.observation_space, action_space=env.action_space, device=device).to(device)
teacher_policy.load_state_dict(weights['policy'])
teacher_policy.eval()

with torch.inference_mode():
    with h5py.File(f"{log_dir}/data.h5df", 'a') as h5_file:
        obs, info = env.reset()
        while simulation_app.is_running():
            for _ in tqdm(range(100)):
                act, _, _ = teacher_policy.compute({'states': teacher_state_preprocessor(obs)})
                next_obs, rew, terminated, truncated, info = env.step(act)
                done = terminated | truncated
                obs_dict = unflatten_tensorized_space(env.observation_space, obs)
                transition = {
                    'obs': {
                        'evader': obs_dict['evader'],
                        'chaser': obs_dict['chaser'] | info
                    },
                    'act': act,
                    'rew': rew,
                    'done': done
                }
                append_dict_to_hdf5(h5_file, transition)
                obs = next_obs
            break
env.close()
