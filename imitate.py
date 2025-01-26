import argparse
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from my_utils.dataset import ILDataset
from models.student import TransformerStudent, TCNStudent
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="Override nested configuration parameters")
parser.add_argument("--device", type=str, required=True)
args_cli = parser.parse_args()
cfg_cli = OmegaConf.create({k: v for k, v in vars(args_cli).items() if v is not None})

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
cfg = OmegaConf.load('configs/transformer_cfg.yaml')
cfg = OmegaConf.merge(cfg, cfg_cli)
cfg['timestamp'] = timestamp
log_dir = f'logs/policy/student/{cfg.model.name}_{timestamp}'
device = cfg.device

model = TransformerStudent(
    cfg.model.n_hist, cfg.model.n_pred,
    seperate_depth=cfg.model.seperate_depth,
    learnable_posemb=cfg.model.learnable_posemb,
    fourier_feature=cfg.model.fourier_feature
).to(device)
if cfg.model.checkpoint:
    model.load_state_dict(torch.load(cfg.model.checkpoint))
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

trainset = ILDataset('data/train', cfg.model.n_hist, cfg.model.n_pred, inverse_depth=cfg.dataset.inverse_depth)
evalset = ILDataset('data/test', cfg.model.n_hist, cfg.model.n_pred, inverse_depth=cfg.dataset.inverse_depth)
trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8) 
evalloader = DataLoader(evalset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)
print(f'Size of train dataset: {len(trainset)}')
print(f'Size of eval dataset: {len(evalset)}')

if cfg.train.loss == 'l1':
    loss_function = F.l1_loss
elif cfg.train.loss == 'smooth_l1':
    loss_function = F.smooth_l1_loss
elif cfg.train.loss == 'mse':
    loss_function = F.mse_loss
else:
    raise ValueError('Undefined loss function')

writer = SummaryWriter(log_dir)
OmegaConf.save(cfg, f'{log_dir}/config.yaml')
total_step = 0
min_eval_loss = float('inf')
for epoch in range(1, cfg.train.epochs + 1):

    print('Epoch', epoch)

    # train
    model.train()
    for data in tqdm(trainloader, desc='Training'):

        # preprocess
        x, y = data
        depth, rgb, _, chaser_state, last_action = x
        depth, rgb, chaser_state, last_action, y = depth.to(device), rgb.to(device), chaser_state.to(device), last_action.to(device), y.to(device)
        
        # optimize
        optimizer.zero_grad()
        y_pred = model(depth, rgb, chaser_state, last_action)
        loss = loss_function(y, y_pred, reduction='sum')
        loss.backward()
        optimizer.step()

        # log
        total_step += 1
        writer.add_scalar(f'train_loss', loss.item() / y.numel(), total_step)

    # eval
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for data in tqdm(evalloader, desc='Evaluating'):
            x, y = data
            depth, rgb, _, chaser_state, last_action = x
            depth, rgb, chaser_state, last_action = depth.to(device), rgb.to(device), chaser_state.to(device), last_action.to(device)
            y = y.to(device)
            y_pred = model(depth, rgb, chaser_state, last_action)
            loss = F.mse_loss(y, y_pred, reduction='sum')
            eval_loss += loss.item()
    
    # log
    writer.add_scalar(f'eval_loss', eval_loss / (len(evalset) * cfg.model.n_pred * 4), epoch)
    torch.save(model.state_dict(), f'{log_dir}/latest_model.pth')
    if eval_loss < min_eval_loss:
        min_eval_loss = eval_loss
        torch.save(model.state_dict(), f'{log_dir}/best_model.pth')