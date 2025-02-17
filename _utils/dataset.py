import os
import torch
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader

class ILDataset(Dataset):
    def __init__(self, data_dir, n_hist, n_pred=1):
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.metadata = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.hdf5'):
                filepath = os.path.join(data_dir, filename)
                with h5py.File(filepath, 'r') as f:
                    len = f['episode_length'].shape[0]
                    num_envs = f['episode_length'].shape[1]
                    for t in range(len - self.n_pred + 1):
                        for env_idx in range(num_envs):
                            if f['done'][t:t+n_pred-1, env_idx, 0].all():
                                self.metadata.append((filepath, t, env_idx))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filepath, t, env_idx = self.metadata[idx]
        with h5py.File(filepath, 'r') as f:

            # TODO
            # 
            
            # observation
            depth = torch.tensor(f['depth'][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 224, 224)
            depth.clamp_max_(10.0)
            depth.div_(10.0)
            rgb = torch.tensor(f['rgb'][t-self.n_hist+1:t+1])  # (H, 224, 224, 3)
            rgb = rgb.permute(0, 3, 1, 2).contiguous()  # (H, 3, 224, 224)
            rgb = rgb.float() / 255.0
            evader_state = torch.tensor(f['evader_state'][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 6)
            chaser_state = torch.tensor(f['chaser_state'][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 15)
            last_action = torch.tensor(f['last_action'][t-self.n_hist+1:t+1], dtype=torch.float32)  # (H, 4)
            
            # action
            action = torch.tensor(f['action'][t:t+self.n_pred], dtype=torch.float32)  # (P, 4)

        return (depth, rgb, evader_state, chaser_state, last_action), action


if __name__ == '__main__':
    dataset = ILDataset('data/train', 10, 5, inverse_depth=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    for batch in dataloader:
        x, y = batch
        depth, rgb, evader_state, chaser_state, last_action = x
        print('x:')
        for tensor in x:
            print(tensor.shape)
        print('y:\n', y.shape)
        break