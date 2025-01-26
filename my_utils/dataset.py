import os
import torch
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader

class ILDataset(Dataset):
    def __init__(self, path, n_hist, n_pred, inverse_depth: bool):
        self.path = path
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.inverse_depth = inverse_depth
        self.chaser_state_amp = torch.tensor(
            [
                0.6350, 0.4774, 0.4190,
                1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                0.8979, 0.6848, 0.5907
            ],
            requires_grad=False
        )
        
        self.metadata = []
        for filename in os.listdir(self.path):
            if filename.endswith(".hdf5"):
                filepath = os.path.join(self.path, filename)
                with h5py.File(filepath, "r") as f:
                    length = f["action"].shape[0]
                    assert length == f["chaser_state"].shape[0] and length == f["depth"].shape[0], "observations and actions are not aligned"
                    for t in range(self.n_hist - 1, length - self.n_pred + 1):
                        self.metadata.append((filepath, t))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filepath, t = self.metadata[idx]
        with h5py.File(filepath, "r") as f:
            
            # observation
            depth = torch.tensor(f["depth"][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 224, 224)
            if self.inverse_depth:
                depth = 1.0 / (1.0 + depth)
            else:
                depth = depth.clamp_max(5.0) / 5.0
            rgb = torch.tensor(f["rgb"][t-self.n_hist+1:t+1])  # (H, 224, 224, 3)
            rgb = rgb.permute(0, 3, 1, 2).contiguous()  # (H, 3, 224, 224)
            rgb = rgb.float() / 255.0
            evader_state = torch.tensor(f["evader_state"][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 6)
            chaser_state = torch.tensor(f["chaser_state"][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 15)
            last_action = torch.tensor(f["last_action"][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 4)
            
            # action
            action = torch.tensor(f["action"][t:t+self.n_pred], dtype=torch.float32)

        return (depth, rgb, evader_state, chaser_state, last_action), action


if __name__ == "__main__":
    dataset = ILDataset("data/train", 10, 5, inverse_depth=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    for batch in dataloader:
        x, y = batch
        depth, rgb, evader_state, chaser_state, last_action = x
        print("x:")
        for tensor in x:
            print(tensor.shape)
        print("y:\n", y.shape)
        break