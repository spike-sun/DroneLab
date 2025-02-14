import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space


class EvaderPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )
        
        self.policy_mean = nn.Sequential(
            nn.Linear(300, self.num_actions),
            nn.Tanh()
        )
        
        self.policy_logstd = nn.Parameter(torch.ones(self.num_actions) * 0.5)

        orthogonal_initialize(self)
    
    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)
    
    def compute(self, inputs, role=None):
        obs = unflatten_tensorized_space(self.observation_space, inputs['states'])
        obs = torch.cat([
            obs['evader']['lidar'].flatten(1,2),
            obs['evader']['height_error'],
            obs['evader']['lin_vel_w_error'],
            obs['evader']['z_axis'],
            obs['evader']['last_action']
        ], dim=-1)
        h = self.net(obs)
        return self.policy_mean(h), self.policy_logstd, {}


class EvaderValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        orthogonal_initialize(self)
    
    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role=None):
        obs = unflatten_tensorized_space(self.observation_space, inputs["states"])
        obs = torch.cat([
            obs['evader']['lidar'].flatten(-2,-1),
            obs['evader']['height_error'],
            obs['evader']['lin_vel_w_error'],
            obs['evader']['z_axis'],
            obs['evader']['last_action']
        ], dim=-1)
        return self.net(obs), {}


def orthogonal_initialize(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)