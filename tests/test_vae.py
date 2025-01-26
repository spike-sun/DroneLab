import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from my_utils.dataset import ILDataset
from models.vae import loss_function, VAE
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = "logs/VAE/VAE_2024-12-19_09-32-20/best_model.pth"
dataset = ILDataset("data/policy/train", 1, 1)
dataloader = DataLoader(dataset, 4, True, num_workers=8)

model = VAE(in_channels=4).to(device)
model.load_state_dict(torch.load(checkpoint))

# eval
model.eval()
eval_loss = 0
with torch.no_grad():
    for data in dataloader:
        x, _ = data
        depth, segmentation, _, _, _ = x
        depth, segmentation = depth.to(device), segmentation.to(device)
        x = torch.concat([segmentation[:,0,:,:,:], depth], dim=1)
        recon_x, mu, logvar = model(x)
        loss = loss_function(x, recon_x, mu, logvar)
        eval_loss += loss.item()

        comparison_rgb = torch.cat([x[0:4, 0:3, :, :], recon_x[0:4, 0:3, :, :]])
        save_image(comparison_rgb.cpu(), "eval_recon_rgb.png", nrow=4)
        img = mpimg.imread("eval_recon_rgb.png")
        plt.figure(figsize=(32, 16))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        
        comparison_d = torch.cat([x[0:4, 3, :, :].unsqueeze(1), recon_x[0:4, 3, :, :].unsqueeze(1)])
        save_image(comparison_d.cpu(), f"eval_recon_d.png", nrow=4)
        img = mpimg.imread("eval_recon_d.png")
        plt.figure(figsize=(32, 16))
        plt.axis('off')
        plt.imshow(img)
        plt.show()