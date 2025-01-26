##########################
#    Launch Isaac Sim    #
##########################

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train/Play an RL agent with skrl.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument('--train', action='store_true', help='Train mode.')
mode_group.add_argument('--play', action='store_true', help='Play mode.')
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, required=True, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#######################
#    Anything else    #
#######################

import os
from datetime import datetime
import torch
from torch import nn
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

import skrl.utils
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import parse_env_cfg

import envs
from models.evader import EvaderPolicy, EvaderValue
from models.teacher import TeacherPolicy, TeacherValue
from configs.skrl_ppo_cfg import PPO_CONFIG
from envs.wrappers import FlattenObservation, IsaacLabWrapper


def main():

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = PPO_CONFIG["agent"]
    trainer_cfg = PPO_CONFIG["trainer"]
    if args_cli.task == 'ForestEvader':
        agent_cfg["experiment"]["directory"] = os.path.join(agent_cfg["experiment"]["directory"], 'evader')
    if args_cli.task == 'ForestChaser':
        agent_cfg["experiment"]["directory"] = os.path.join(agent_cfg["experiment"]["directory"], 'teacher')

    log_root_path = os.path.abspath(agent_cfg["experiment"]["directory"])
    if args_cli.train:
        agent_cfg["experiment"]["experiment_name"] += datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, agent_cfg["experiment"]["experiment_name"])
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), PPO_CONFIG)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), PPO_CONFIG)
    if args_cli.play:
        assert args_cli.checkpoint, 'please specify a checkpoint'
        resume_path = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set seed
    if args_cli.seed is not None:
        seed = args_cli.seed
        print(f'using cli seed {seed}')
    else:
        seed = agent_cfg["experiment"]["seed"]
        if env_cfg.seed != seed:
            print(f'WARNING: env seed ({env_cfg.seed}) and skrl seed ({seed}) are different, using skrl seed')
    env_cfg.seed = seed
    skrl.utils.set_seed(seed)
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train" if args_cli.train else "play"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = RecordVideo(env, **video_kwargs)
    env = IsaacLabWrapper(env)
    env = FlattenObservation(env)
    if args_cli.task == 'ForestEvader':
        models= {
            "policy": EvaderPolicy(env.observation_space, env.action_space, device=env.unwrapped.device),
            "value": EvaderValue(env.observation_space, 1, device=env.unwrapped.device)
        }
    
    elif args_cli.task == 'ForestChaser':
        models = {
            "policy": TeacherPolicy(env.observation_space, env.action_space, device=env.unwrapped.device),
            "value": TeacherValue(env.observation_space, 1, device=env.unwrapped.device)
        }
    
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.flatten_observation_space, "device": env.unwrapped.device}
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.unwrapped.device}

    if args_cli.train:
        agent = PPO(
            models=models,
            memory=RandomMemory(memory_size=agent_cfg["rollouts"], num_envs=env.num_envs, device=env.unwrapped.device),
            cfg=agent_cfg,
            observation_space=env.flatten_observation_space,
            action_space=env.action_space,
            device=env.unwrapped.device
        )
        if args_cli.checkpoint:
            agent.load(args_cli.checkpoint)
            #models['policy'].policy_logstd = nn.Parameter(torch.ones(gym.spaces.flatdim(agent.action_space), device=env.unwrapped.device) * 0.5)
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
        trainer.train()
    
    if args_cli.play:
        agent_cfg["experiment"]["write_interval"] = 0       # don't log to Tensorboard
        agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        agent_cfg["random_timesteps"] = 0                   # ignore random timesteps
        agent = PPO(
            models=models,
            memory=None,
            cfg=agent_cfg,
            observation_space=env.flatten_observation_space,
            action_space=env.action_space,
            device=env.unwrapped.device
        )
        agent.init()
        print(f"[INFO] Loading model checkpoint from {resume_path}")
        agent.load(resume_path)
        agent.set_running_mode("eval")
        obs, _ = env.reset()
        timestep = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                actions, logprobs, outputs = agent.act(obs, timestep=0, timesteps=0)
                obs, reward, terminated, truncated, info = env.step(outputs["mean_actions"])
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()