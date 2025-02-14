from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from _utils.dataset import ILDataset
from models.student import TransformerStudent
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16})
import numpy as np

device = "cuda:0"
n_hist = 10
n_pred = 5

dataset = ILDataset("data/policy/test", n_hist, n_pred)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
model = TransformerStudent(n_hist, n_pred).to(device)
model.load_state_dict(torch.load("logs/policy/TransformerStudent_2024-12-26_15-18-54/best_model.pth", weights_only=True))
n = len(dataset)
errors = np.zeros((n, n_pred, 4))
model.eval()
total_error = torch.zeros(n_pred, 4, device=device)
i = 0
with torch.no_grad():
    for data in tqdm(dataloader, desc="Testing"):
        x, y = data
        depth, segmentation, evader_state, chaser_state, last_action = x
        depth, segmentation, chaser_state, last_action = depth.to(device), segmentation.to(device), chaser_state.to(device), last_action.to(device)
        x = (depth, segmentation, evader_state, chaser_state, last_action)
        y = y.to(device)  # (1, P, 4)
        y_pred = model(x)  # (1, P, 4)
        y = y.squeeze()  # (P, 4)
        y_pred = y_pred.squeeze()  # (P, 4)
        errors[i, :, :] = (y_pred - y).cpu().numpy()
        i += 1

np.save("logs/errors.npy", errors)

se = errors * errors
print(f"MSE={np.mean(se):.6f}")
print(f"Std={np.std(se):.6f}")

fig, axes = plt.subplots(4, 1, figsize=(8, 4 * 4))
t = np.linspace(1, n_pred, n_pred)
for d in range(4):
    mean = np.mean(se[:, :, d], axis=0)
    std = np.std(se[:, :, d], axis=0)
    axes[d].errorbar(t, mean, yerr=std, fmt='-o', capsize=5, label='Data with error bars')
    axes[d].set_title(f'Dimension {d+1}')
    axes[d].set_xlabel('t')
    axes[d].set_ylabel('MSE')
    axes[d].set_ylim([-0.0005, 0.002])
    axes[d].grid(True)

plt.tight_layout()
plt.show()