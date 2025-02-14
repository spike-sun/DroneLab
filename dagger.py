##########################
#    Launch Isaac Sim    #
##########################

import os
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
os.environ['ENABLE_CAMERAS'] = '1'
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#######################
#    Anything else    #
#######################

from omegaconf import OmegaConf
import torch
from envs.forest_chaser import ForestChaserCfg, ForestChaser
from envs.wrappers import FlattenObservation
from models.student import TransformerStudent
from my_utils.tensor_queue import TensorQueue

def get_chaser_input(observation):
    depth = observation['chaser']['depth']
    depth = depth.clamp_max(5.0) / 5.0
    rgb = observation['chaser']['rgb']
    rgb = rgb.permute(0, 3, 1, 2).contiguous()
    rgb = rgb.float() / 255.0
    chaser_state = torch.cat([
        observation['chaser']['lin_vel_b'],
        observation['chaser']['rotmat'],
        observation['chaser']['ang_vel_b']
    ], dim=-1)
    last_action = observation['chaser']['last_action']
    return depth, rgb, chaser_state, last_action


def main():
    device = args_cli.device
    num_envs = args_cli.num_envs

    cfg = ForestChaserCfg()
    cfg.scene.num_envs = num_envs
    env = ForestChaser(cfg)

    # chaser
    cfg = OmegaConf.load('logs/policy/student/Transformer_2025-01-13_21-06-52/config.yaml')
    student_policy = TransformerStudent(
        cfg.model.n_hist, cfg.model.n_pred,
        seperate_depth=cfg.model.seperate_depth,
        learnable_posemb=cfg.model.learnable_posemb,
        fourier_feature=cfg.model.fourier_feature
    ).to(device)
    student_policy.load_state_dict(torch.load("logs/policy/student/Transformer_2025-01-13_21-06-52/best_model.pth", weights_only=True))
    student_policy.eval()
    depth_buffer = TensorQueue(device, env.num_envs, cfg.model.n_hist, 224, 224)
    rgb_buffer = TensorQueue(device, env.num_envs, cfg.model.n_hist, 3, 224, 224)
    chaser_state_buffer = TensorQueue(device, env.num_envs, cfg.model.n_hist, 15)
    last_action_buffer = TensorQueue(device, env.num_envs, cfg.model.n_hist, 4)

    observation, info = env.reset()
    depth, rgb, chaser_state, last_action = get_chaser_input(observation)
    depth_buffer.init(depth)
    rgb_buffer.init(rgb)
    chaser_state_buffer.init(chaser_state)
    last_action_buffer.init(last_action)

    while simulation_app.is_running():
        
        depth, rgb, chaser_state, last_action = get_chaser_input(observation)
        depth_buffer.append(depth)
        rgb_buffer.append(rgb)
        chaser_state_buffer.append(chaser_state)
        last_action_buffer.append(last_action)
        
        with torch.inference_mode():
            chaser_action = student_policy(depth_buffer.buffer, rgb_buffer.buffer, chaser_state_buffer.buffer, last_action_buffer.buffer)

        obs, rew, terminated, truncated, info = env.step(chaser_action[:, 0])
    
    env.close()

if __name__ == '__main__':
    main()