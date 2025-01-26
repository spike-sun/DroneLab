import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary

class VAEEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_convs = nn.Sequential(
            self.conv_block(in_channels, 16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(16, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv_block(128, 512)
        )
        self.initialize_weights()
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_convs(x)  # (B, 512, 14, 14)


class VAE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # Encoder
        self.encoder_convs = VAEEncoder(in_channels)

        # Decoder
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            self.conv_block(256, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            self.conv_block(128, 64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            self.conv_block(64, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            self.conv_block(32, 16),
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def encode(self, x):
        h = self.encoder_convs(x)  # (B, 512, 14, 14)
        z_mean = h[:, :256, :, :]
        z_logvar = h[:, 256:, :, :]
        return z_mean, z_logvar
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_convs(z)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)  # (B, 256, 14, 14)
        z = self.sample(mu, logvar)  # (B, 256, 14, 14)
        return self.decode(z), mu, logvar


def vae_loss(x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
    mse_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = - 0.5 * torch.sum(1 + logvar - mu * mu - logvar.exp())
    return mse_loss + kld_loss


if __name__ == "__main__":
    model = VAEEncoder(4)
    summary(model.cuda(), (4, 224, 224))