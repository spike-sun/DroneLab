##########################
#    Launch Isaac Sim    #
##########################

import os
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train/Play an RL agent with skrl.")
mode_group = parser.add_mutually_exclusive_group(required=True)
mode_group.add_argument('--train', action='store_true')
mode_group.add_argument('--play', action='store_true')
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--use_fabric", action="store_true", default=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ['ENABLE_CAMERAS'] = str(int(args_cli.enable_cameras))
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#######################
#    Anything else    #
#######################

import os
from datetime import datetime
import torch
import gymnasium as gym

import skrl.utils
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import IsaacLabWrapper
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer

from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg

import envs
from models.evader import EvaderPolicy, EvaderValue
from models.teacher import TeacherPolicy, TeacherValue
from configs.skrl_ppo_cfg import PPO_CONFIG


def main():

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=args_cli.use_fabric)
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
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), PPO_CONFIG)
    if args_cli.play:
        assert args_cli.checkpoint, 'must specify a checkpoint'
        resume_path = os.path.abspath(args_cli.checkpoint)
        log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set seed
    if args_cli.seed is not None:
        seed = args_cli.seed
        print(f'[INFO] using cli seed {seed}')
    else:
        seed = agent_cfg["experiment"]["seed"]
        if env_cfg.seed != seed:
            print(f'[WARNING] env seed ({env_cfg.seed}) and skrl seed ({seed}) are different, using skrl seed')
    env_cfg.seed = seed
    skrl.utils.set_seed(seed)
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabWrapper(env)
    if args_cli.task == 'ForestEvader':
        models= {
            "policy": EvaderPolicy(env.observation_space, env.action_space, device=env.device),
            "value": EvaderValue(env.observation_space, 1, device=env.device)
        }
    
    elif args_cli.task == 'ForestChaser':
        models = {
            "policy": TeacherPolicy(env.observation_space, env.action_space, device=env.device),
            "value": TeacherValue(env.observation_space, 1, device=env.device)
        }
    
    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
    agent_cfg["learning_rate_scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR
    agent_cfg["learning_rate_scheduler_kwargs"] = {
        "T_max": trainer_cfg["timesteps"] // agent_cfg["rollouts"] * agent_cfg["learning_epochs"],
        "eta_min": 1e-4
    }

    if args_cli.train:
        agent = PPO(
            models=models,
            memory=RandomMemory(memory_size=agent_cfg["rollouts"], num_envs=env.num_envs, device=env.device),
            cfg=agent_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device
        )
        if args_cli.checkpoint:
            agent.load(args_cli.checkpoint)
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
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device
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
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()