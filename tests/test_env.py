##########################
#    Launch Isaac Sim    #
##########################

import os
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--use_fabric", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ['ENABLE_CAMERAS'] = str(int(args_cli.enable_cameras))
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


#######################
#    Anything else    #
#######################

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import envs


def main():
    device = args_cli.device
    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=args_cli.use_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    observation, info = env.reset()
    while simulation_app.is_running():
        action = torch.zeros((num_envs, gym.spaces.flatdim(env.action_space)), device=device)
        observation, reward, terminated, truncated, info = env.step(action)
    env.close()

if __name__ == '__main__':
    main()

'''
from omni.isaac.core.articulations import ArticulationView
prims = ArticulationView(prim_paths_expr="/World/envs/env.*/evader", name="cf2x")
prims.initialize()
print(prims.get_body_masses())
print(prims.get_body_inertias())
tensor([[0.0250, 0.0008, 0.0008, 0.0008, 0.0008]], device='cuda:0')
tensor([[[1.6572e-05, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 1.6656e-05, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 2.9262e-05],
         [2.0000e-09, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 1.6700e-07, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 1.6800e-07],
         [2.0000e-09, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 1.6700e-07, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 1.6800e-07],
         [2.0000e-09, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 1.6700e-07, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 1.6800e-07],
         [2.0000e-09, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 1.6700e-07, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 1.6800e-07]]], device='cuda:0')
'''