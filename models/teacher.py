import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.spaces.torch import unflatten_tensorized_space, flatten_tensorized_space


class TeacherPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.backbone = nn.Sequential(
            nn.Linear(79, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )
        self.policy_mean = nn.Sequential(
            nn.Linear(300, self.num_actions),
            nn.Tanh()
        )
        self.policy_logstd = nn.Parameter(torch.ones(4) * 0.5)
        orthogonal_initialize(self)
    
    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role=None):
        obs = unflatten_tensorized_space(self.observation_space, inputs['states'])
        obs = torch.concat([
            obs['chaser']["lidar"].flatten(-2,-1),
            obs['chaser']['relative_pos'],
            obs['chaser']['relative_vel'],
            obs['chaser']['lin_vel_b'],
            obs['chaser']['rotmat'],
            obs['chaser']['ang_vel_b'],
            obs['chaser']['last_action'],
            obs['evader']['z_axis'],
            obs['evader']['last_action']
        ], dim=-1).to(self.device)
        h = self.backbone(obs)
        return self.policy_mean(h), self.policy_logstd, {}


class TeacherValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.value = nn.Sequential(
            nn.Linear(79, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        orthogonal_initialize(self)
    
    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role=None):
        obs = unflatten_tensorized_space(self.observation_space, inputs['states'])
        obs = torch.concat([
            obs['chaser']["lidar"].flatten(-2,-1),
            obs['chaser']['relative_pos'],
            obs['chaser']['relative_vel'],
            obs['chaser']['lin_vel_b'],
            obs['chaser']['rotmat'],
            obs['chaser']['ang_vel_b'],
            obs['chaser']['last_action'],
            obs['evader']['z_axis'],
            obs['evader']['last_action']
        ], dim=-1).to(self.device)
        return self.value(obs), {}


class TeacherShared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")
        self.backbone = nn.Sequential(
            nn.Linear(79, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.LayerNorm(400),
            nn.ReLU()
        )
        self.policy_mean = nn.Sequential(
            nn.Linear(400, 4),
            nn.Tanh()
        )
        self.policy_logstd = nn.Parameter(torch.zeros(4))
        self.value = nn.Linear(400, 1)
        self.h = None
        orthogonal_initialize(self)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        if role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self.h = self.backbone(inputs["states"])
            return self.policy_mean(self.h), self.policy_logstd, {}
        if role == "value":
            if self.h is not None:
                h = self.h
                self.h = None
            else:
                h = self.backbone(inputs["states"])
            return self.value(h), {}

def orthogonal_initialize(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)