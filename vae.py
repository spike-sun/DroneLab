from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torch.utils.tensorboard.writer import SummaryWriter
from _utils.dataset import ILDataset
from models.vae import vae_loss, VAE


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log_dir = "logs/VAE/VAE" + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir)
    
    BATCH_SIZE = 120
    EPOCHS = 20
    checkpoint = None

    dataset = ILDataset("data/train", 1, 1)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    dataset_train, dataset_eval = random_split(dataset, [train_size, eval_size])
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    dataloader_eval = DataLoader(dataset_eval, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    print(f'Size of train dataset: {train_size}')
    print(f'Size of eval dataset: {eval_size}')

    # configure
    model = VAE(in_channels=4).to(device)
    if(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    min_eval_loss = float('inf')
    total_timestep = 0
    for epoch in range(1, EPOCHS + 1):
        
        print("Epoch", epoch)

        # train
        model.train()
        train_loss = 0
        for data in tqdm(dataloader_train, desc="Training"):
            
            x, _ = data
            depth, segmentation, _, _, _ = x
            depth, segmentation = depth.to(device), segmentation.to(device)
            x = torch.concat([segmentation[:,0,:,:,:], depth], dim=1)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(x, recon_x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_timestep += 1
            writer.add_scalar(f"train_loss", loss.item(), total_timestep)
            
        comparison_rgb = torch.cat([x[0:4, 0:3, :, :], recon_x[0:4, 0:3, :, :]])
        save_image(comparison_rgb.cpu(), f'{log_dir}/train_recon_rgb.png', nrow=4)
        comparison_d = torch.cat([x[0:4, 3, :, :].unsqueeze(1), recon_x[0:4, 3, :, :].unsqueeze(1)])
        save_image(comparison_d.cpu(), f'{log_dir}/train_recon_d.png', nrow=4)

        # eval
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for data in tqdm(dataloader_eval, desc="Evaluating"):
                x, _ = data
                depth, segmentation, _, _, _ = x
                depth, segmentation = depth.to(device), segmentation.to(device)
                x = torch.concat([segmentation[:,0,:,:,:], depth], dim=1)
                recon_x, mu, logvar = model(x)
                loss = vae_loss(x, recon_x, mu, logvar)
                eval_loss += loss.item()
            writer.add_scalar(f"eval_loss", eval_loss / len(dataloader_eval), epoch)
            comparison_rgb = torch.cat([x[0:4, 0:3, :, :], recon_x[0:4, 0:3, :, :]])
            save_image(comparison_rgb.cpu(), f'{log_dir}/eval_recon_rgb.png', nrow=4)
            comparison_d = torch.cat([x[0:4, 3, :, :].unsqueeze(1), recon_x[0:4, 3, :, :].unsqueeze(1)])
            save_image(comparison_d.cpu(), f'{log_dir}/eval_recon_d.png', nrow=4)
        
        # save checkpoint
        torch.save(model.state_dict(), f'{log_dir}/latest_model.pth')
        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            torch.save(model.state_dict(), f'{log_dir}/best_model.pth')