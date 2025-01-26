from typing import Union
import torch
import torch.nn as nn
import gymnasium as gym


class RunningStandardScaler(nn.Module):
    def __init__(self, size: Union[int, gym.Space], epsilon: float = 1e-8, clip_threshold: float = 10.0) -> None:
        '''Standardize the input data by removing the mean and scaling by the standard deviation'''
        super().__init__()
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        if isinstance(size, gym.Space):
            size = gym.spaces.flatdim(size)
        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float, requires_grad=False))
        self.register_buffer("running_var", torch.ones(size, dtype=torch.float, requires_grad=False))
        self.register_buffer("current_size", torch.ones(1, dtype=torch.float, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if self.training:
                # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
                batch_mean = torch.mean(x, dim=0)
                batch_var = torch.var(x, dim=0)
                batch_size = float(x.shape[0])
                delta = batch_mean - self.running_mean
                total_size = self.current_size + batch_size
                moment = self.running_var * self.current_size + batch_var * batch_size + self.current_size * batch_size / total_size * delta ** 2
                self.running_mean = self.running_mean + batch_size / total_size * delta
                self.running_var = moment / total_size
                self.current_size = total_size
            x_normalized = (x - self.running_mean) / (torch.sqrt(self.running_var) + self.epsilon)
            return x_normalized.clamp(-self.clip_threshold, self.clip_threshold)


if __name__ == '__main__':
    size = 3
    scalar = RunningStandardScaler(size)
    print(scalar.running_mean)
    print(scalar.running_var)
    print(scalar.current_size)

    scalar.train()
    x1 = torch.rand((100, size))
    x1 = scalar(x1)
    print(x1.mean(dim=0), x1.var(dim=0))
    print(scalar.running_mean)
    print(scalar.running_var)
    print(scalar.current_size)

    scalar.eval()
    x2 = torch.rand((100, size))
    x2 = scalar(x2)
    print(scalar.running_mean)
    print(scalar.running_var)
    print(scalar.current_size)