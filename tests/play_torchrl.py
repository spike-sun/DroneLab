##########################
#    Launch Isaac Sim    #
##########################

import os
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, required=True, help="Number of environments to simulate.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
os.environ['ENABLE_CAMERAS'] = str(int(args_cli.enable_cameras))
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#######################
#    Anything else    #
#######################

from datetime import datetime
import torch
from torch import nn
import gymnasium as gym
import matplotlib.pyplot as plt

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import GymWrapper, TransformedEnv, StepCounter, VecNorm, IsaacGymWrapper, step_mdp
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.ppo import ClipPPOLoss
from tqdm import tqdm
import envs
from isaaclab_tasks.utils import parse_env_cfg
import copy
from torch.utils.tensorboard import SummaryWriter


class IsaacLabWrapper(GymWrapper):
    def __init__(self, env=None, categorical_action_encoding=False, **kwargs):
        super().__init__(env, categorical_action_encoding, **kwargs)
    def read_action(self, action: torch.Tensor):
        return action.detach()
    def read_done(self, terminated: torch.Tensor, truncated: torch.Tensor, done: torch.Tensor):
        done = terminated | truncated
        return terminated.clone(), truncated.clone(), done, done.any()
    def read_obs(self, observations: dict):
        return copy.deepcopy(observations)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x: TensorDict):
        x = torch.cat([
            x['lidar'].flatten(-2,-1),
            x['height_error'],
            x['lin_vel_w_error'],
            x['z_axis'],
            x['last_action']
        ], dim=-1)
        x = self.net(x)
        return x


# parameters
log_dir = datetime.now().strftime("logs/torchrl/%Y-%m-%d_%H-%M-%S")
device = args_cli.device
lr = 5e-4
max_grad_norm = 1.0
rollout = 256
frames_per_batch = args_cli.num_envs * rollout
epoch = 1000
total_frames = frames_per_batch * epoch
batch_size = torch.Size([args_cli.num_envs])
mini_batch_size = 2048  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 1  # optimisation steps per batch of data collected
clip_epsilon = 0.2  # clip value for PPO loss: see the equation in the intro for more context.
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

scalar_ckpt = torch.load("logs/torchrl/2025-02-12_17-03-35/checkpoint/best_scalar.pt")
scalar_ckpt['_extra_state'] = scalar_ckpt.pop('transforms.1._extra_state')
vec_norm = VecNorm()
vec_norm.load_state_dict(scalar_ckpt)
vec_norm.eval()
vec_norm.freeze()

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
env = gym.make(args_cli.task, cfg=env_cfg)
env = IsaacLabWrapper(env, device=device)
env = TransformedEnv(env, StepCounter(max_steps=1800))
env = TransformedEnv(env, vec_norm)
check_env_specs(env)
print(env.observation_spec)
print(env.action_spec)

policy_ckpt = torch.load("logs/torchrl/2025-02-12_17-03-35/checkpoint/best_policy.pt")
policy_net = nn.Sequential(
    MLP(input_size=58, output_size=6),
    NormalParamExtractor()
).to(device)
policy_module = ProbabilisticActor(
    module = TensorDictModule(policy_net, in_keys=[("observation", "evader")], out_keys=["loc", "scale"]),
    spec = env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class = TanhNormal,
    distribution_kwargs = {
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob = True
)
policy_module.load_state_dict(policy_ckpt)
policy_module.eval()

td = env.reset()
while simulation_app.is_running():
    td = policy_module(td)
    _, td = env.step_and_maybe_reset(td)