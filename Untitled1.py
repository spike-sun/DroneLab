import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


def visualize_dataset():

    from my_utils.dataset import ILDataset

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
    from my_utils.dataset import ILDataset
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


if __name__ == "__main__":
    see_checkpoint()