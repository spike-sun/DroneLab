import torch

class CircleTrajectory:
    def __init__(self, num_envs, env_origins, device):
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.device = device
        self.zeros = torch.zeros(num_envs, device=device)
        self.ones = torch.ones(num_envs, device=device)
        self.angvel_min = -0.2
        self.angvel_max = 0.2
        self.reset()

    def reset(self):
        self.t = torch.zeros(self.num_envs, device=self.device)
        self.angvel = torch.rand(self.num_envs, device=self.device) * (self.angvel_max - self.angvel_min) + self.angvel_min
    
    def reset_idx(self, idx):
        self.t[idx] = 0.0
        self.angvel[idx] = torch.rand(len(idx), device=self.device) * (self.angvel_max - self.angvel_min) + self.angvel_min
    
    def step(self, dt):
        w = self.angvel
        theta = w * self.t
        c = torch.cos(theta)
        s = torch.sin(theta)
        pos_des = torch.stack([c - 1.0, s, self.ones * 2.0], dim=-1) + self.env_origins[:, 0:3]
        vel_w_des = torch.stack([- w * s, w * c, self.zeros], dim=-1)
        acc_w_des = torch.stack([- w * w * c, - w * w * s, self.zeros], dim=-1)
        self.t += dt
        return pos_des, vel_w_des, acc_w_des


class EightTrajectory:
    def __init__(self, num_envs, env_origins, R, device):
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.device = device
        self.zeros = torch.zeros(num_envs, device=device)
        self.ones = torch.ones(num_envs, device=device)
        self.angvel_min = 0.08
        self.angvel_max = 0.10
        self.t = torch.zeros(self.num_envs, device=self.device)
        self.angvel = torch.ones(self.num_envs, device=self.device) * 0.1
        self.R = R
        self.reset()

    def reset(self):
        self.reset_idx(range(self.num_envs))
    
    def reset_idx(self, idx):
        self.t[idx] = 0.0
        angvel = torch.rand(len(idx), device=self.device) * (self.angvel_max - self.angvel_min) + self.angvel_min
        self.angvel[idx] = angvel
    
    def step(self, dt):
        w = self.angvel
        theta = w * self.t + 3.1415926 / 2.0
        c = torch.cos(theta)
        s = torch.sin(theta)
        c2 = torch.cos(2 * theta)
        s2 = torch.sin(2 * theta)
        pos_des = torch.stack([self.R * c, 0.5 * self.R * s2, self.ones * 2.0], dim=-1) + self.env_origins[:, 0:3]
        vel_w_des = torch.stack([- self.R * w * s, self.R * w * c2, self.zeros], dim=-1)
        acc_w_des = torch.stack([- self.R * w * w * c, - 2.0 * self.R * w * w * s2, self.zeros], dim=-1)
        self.t += dt
        return pos_des, vel_w_des, acc_w_des
