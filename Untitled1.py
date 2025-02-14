import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import h5py
import ale_py
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ListStorage
from torchrl.envs import GymEnv, GymWrapper


def visualize_dataset():

    from _utils.dataset import ILDataset

    dataset = ILDataset("data/42", 1, 1, inverse_depth=True)

    plt.ion()
    fig, ax = plt.subplots(1,2)
    plt.tight_layout()

    for i in range(len(dataset)):
        x, y = dataset[i]
        depth, rgb, evader_state, chaser_state, last_action = x
        depth = depth.squeeze().numpy()
        rgb = rgb.squeeze().permute(1,2,0).numpy()
        
        ax[0].imshow(depth, cmap='gray', vmin=0, vmax=1)
        ax[1].imshow(rgb)
        ax[0].axis('off')
        ax[1].axis("off")
        plt.draw()
        plt.pause(0.001)
        ax[0].clear()
        ax[1].clear()

    plt.ioff()
    plt.close(fig)


def statistics():
    from _utils.dataset import ILDataset
    # 统计state最大值最小值
    dataset = ILDataset("data/train", 1, 1, inverse_depth=False)
    n = len(dataset)
    vmax = torch.zeros(19) - float("inf")
    for i in range(n):
        x, y = dataset[i]
        depth, rgb, evader_state, chaser_state, last_action = x
        state = torch.concat([chaser_state, last_action], dim=1).squeeze()
        vmax = torch.maximum(vmax, torch.abs(state))
    print(vmax)


def see_checkpoint():
    module = torch.load('logs/policy/evader/PPO_2025-01-13_17-44-56/checkpoints/best_agent.pt')
    print(module)

def print_h5_structure(obj, name="", indent=0):
    """
    递归打印 HDF5 文件的组结构和数据集形状。

    :param name: 当前对象的名字
    :param obj: 当前对象（Group 或 Dataset）
    :param indent: 当前缩进级别
    """
    space = '  ' * indent  # 定义缩进量
    if isinstance(obj, h5py.Dataset):
        print(f"{space}{name} {obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"{space}{name}/")
        for key, item in obj.items():
            print_h5_structure(item, f"{key}", indent+1)


if __name__ == "__main__":
    with h5py.File(f"data/dagger/data.h5df") as h5_file:
        print_h5_structure(h5_file)