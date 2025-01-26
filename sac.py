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
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
from skrl.agents.torch.sac import SAC
from skrl.memories.torch import RandomMemory
# import the preprocessor class
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import IsaacLabWrapper
from skrl.utils import set_seed
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import parse_env_cfg
from models import MLPCritic, MLPActor
from configs.skrl_sac_cfg import SAC_CONFIG
import envs


def main():

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    agent_cfg = SAC_CONFIG["agent"]
    trainer_cfg = SAC_CONFIG["trainer"]
    experiment_cfg = SAC_CONFIG["agent"]["experiment"]

    set_seed(args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"])
    
    # specify directory for logging experiments
    log_root_path = experiment_cfg["directory"]
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.train:
        experiment_name = experiment_cfg["experiment_name"] + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, experiment_name)
        agent_cfg["experiment"]["experiment_name"] = experiment_name
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    if args_cli.play:
        assert args_cli.checkpoint
        resume_path = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))

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

    # initialize models
    models = {
        "policy": MLPActor(env.observation_space, env.action_space, device=env.device),
        "critic_1": MLPCritic(env.observation_space, env.action_space, device=env.device),
        "critic_2": MLPCritic(env.observation_space, env.action_space, device=env.device),
        "target_critic_1": MLPCritic(env.observation_space, env.action_space, device=env.device),
        "target_critic_2": MLPCritic(env.observation_space, env.action_space, device=env.device)
    }
    
    # setup state/value scaler
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

    if args_cli.train:
        memory = RandomMemory(memory_size=1000000, num_envs=env.num_envs, device=env.device)
        agent = SAC(
            models=models,
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            cfg=agent_cfg
        )
        if args_cli.checkpoint:
            agent.load(args_cli.checkpoint)
        agent.set_running_mode("train")
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
        trainer.train()
    
    if args_cli.play:
        agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
        agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
        agent_cfg["random_timesteps"] = 0  # ignore random timesteps
        agent = SAC(
            models=models,
            memory=None,
            cfg=agent_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )
        agent.init()
        print(f"[INFO] Loading model checkpoint from {resume_path}")
        agent.load(resume_path)
        agent.set_running_mode("eval")
        obs, _ = env.reset()
        timestep = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                _, _, outputs = agent.act(obs, timestep=0, timesteps=0)
                obs, _, _, _, _ = env.step(outputs["mean_actions"])
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()