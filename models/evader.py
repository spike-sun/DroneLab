import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from envs.wrappers import unflatten_observation


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

        self.initialize_weights()
    
    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)
    
    def compute(self, inputs, role=''):
        obs = inputs['states']
        if isinstance(obs, torch.Tensor):
            obs = unflatten_observation(obs, self.observation_space)
        obs = torch.cat([
            obs['history_lidar'].flatten(1,2),
            obs['height_error'],
            obs['lin_vel_w_error'],
            obs['z_axis'],
            obs['last_action']
        ], dim=-1)
        h = self.net(obs)
        return self.policy_mean(h), self.policy_logstd, {}

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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

        self.initialize_weights()
    
    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs: dict, role):
        obs = unflatten_observation(inputs["states"], self.observation_space)
        obs = torch.cat([
            obs['history_lidar'].flatten(1,2),
            obs['height_error'],
            obs['lin_vel_w_error'],
            obs['z_axis'],
            obs['last_action']
        ], dim=-1)
        return self.net(obs), {}

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)