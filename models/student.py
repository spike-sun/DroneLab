import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from .vae import VAEEncoder
LOG10000 = 9.210340372


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, normalized_shape, 1))
        self.beta = nn.Parameter(torch.zeros(1, normalized_shape, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)        
        return self.gamma * x_normalized + self.beta


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, sigma=1.0):
        super().__init__()
        assert out_channels % 2 == 0, "out_channels must be even"
        # 方差越大高频分量越多
        self.register_buffer("W", 2 * torch.pi * sigma * torch.randn(in_channels, out_channels // 2))
        self.register_buffer("scaling_factor", torch.sqrt(torch.tensor(1.0 / out_channels)))

    def forward(self, x):
        z = torch.einsum("bti,ij->btj", x, self.W)
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1) * self.scaling_factor


class TCNStudent(nn.Module):
    def __init__(self, n_hist, n_pred):
        super().__init__()
        self.n_hist = n_hist
        self.n_pred = n_pred

        self.rgbd_tokenizer = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            *list(torchvision.models.resnet18().children())[1:-1]
        )

        self.proj_state = nn.Linear(19, 512)

        self.tcn = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=2, stride=1, dilation=1),
            LayerNorm(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=1, dilation=2),
            LayerNorm(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=1, dilation=2),
            LayerNorm(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, dilation=1),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_pred * 4)
        )

    def forward(self, x) -> torch.Tensor:
        
        # (B, H, 224, 224), (B, H, 3, 224, 224), (B, H, 6), (B, H, 15), (B, H, 4)
        depth, rgb, _, chaser_state, last_action = x
        
        # rgbd
        rgb = rgb.view(-1, 3, 224, 224)  # (B*H, 3, 224, 224)
        depth = depth.view(-1, 1, 224, 224)  # (B*H, 224, 224)
        rgbd = torch.concat([rgb, depth], dim=1)  # (B*H, 4, 224, 224)
        rgbd_token = self.rgbd_tokenizer(rgbd).view(-1, self.n_hist, 512)  # (B, H, 512)
        
        # state
        state = torch.concat([chaser_state, last_action], dim=-1)  # (B, H, 19)
        state_token = self.proj_state(state)  # (B, H, 512)
        
        # merge
        h = rgbd_token + state_token  # (B, H, 512)
        h = h.permute(0, 2, 1)  # (B, 512, H)
        tcn_out = self.tcn(h).squeeze()  # (B, 512)
        action = self.output_layer(tcn_out).view(-1, self.n_pred, 4)  # (B, P*4)

        return action
    

class TransformerStudent(nn.Module):
    def __init__(self, n_hist, n_pred, seperate_depth: bool, learnable_posemb: bool, fourier_feature: bool):
        super().__init__()
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.seperate_depth = seperate_depth
        self.learnable_posemb = learnable_posemb
        self.fourier_feature = fourier_feature
        self.d_model = 512

        if self.seperate_depth:
            self.rgb_tokenizer = nn.Sequential(*list(torchvision.models.resnet18().children())[0:-1])
            self.depth_tokenizer = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *list(torchvision.models.resnet18().children())[1:-1]
            )
            self.n_modals = 3
        else:
            self.rgbd_tokenizer = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                *list(torchvision.models.resnet18().children())[1:-1]
            )
            self.n_modals = 2

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True),
            num_layers=4
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8, batch_first=True),
            num_layers=4
        )

        self.readout = nn.Parameter(torch.zeros(1, self.n_pred, self.d_model))

        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Tanh()
        )

        if self.fourier_feature:
            self.proj_state = RandomFourierFeatures(19, self.d_model)
        else:
            self.proj_state = nn.Linear(19, self.d_model)

        position_encoding = torch.zeros(self.n_hist, self.d_model)
        position = torch.arange(0, self.n_hist, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, self.d_model, 2).float() * LOG10000 / self.d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.repeat(self.n_modals, 1)
        
        if self.learnable_posemb:
            self.position_encoding = nn.Parameter(position_encoding)
        else:
            self.register_buffer('position_encoding', position_encoding)

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(n_pred)
        self.register_buffer("tgt_mask", tgt_mask)
    
    def forward(self, depth, rgb, chaser_state, last_action):
        
        # (B, H, 224, 224), (B, H, 3, 224, 224), (B, H, 15), (B, H, 4)

        state = torch.concat([chaser_state, last_action], dim=-1)  # (B, H, 19)
        state_token = self.proj_state(state)  # (B, H, C)

        rgb = rgb.view(-1, 3, 224, 224)  # (B*H, 3, 224, 224)
        depth = depth.view(-1, 1, 224, 224)  # (B*H, 1, 224, 224)
        if self.seperate_depth:
            rgb_token = self.rgb_tokenizer(rgb).view(-1, self.n_hist, self.d_model)  # (B, H, C)
            depth_token = self.depth_tokenizer(depth).view(-1, self.n_hist, self.d_model)  # (B, H, C)
            src = torch.cat([
                rgb_token,
                depth_token,
                state_token
            ], dim=1) + self.position_encoding  # (B, 3*H, C)
        else:
            rgbd = torch.concat([rgb, depth], dim=1)  # (B*H, 4, 224, 224)
            rgbd_token = self.rgbd_tokenizer(rgbd).view(-1, self.n_hist, self.d_model)  # (B, H, C)
            src = torch.cat([
                rgbd_token,
                state_token
            ], dim=1) + self.position_encoding  # (B, 2*H, C)
        
        memory = self.encoder(src)  # (B, M*H, C)
        tgt = self.readout.expand(memory.shape[0], -1, -1)  # (B, P, C)
        action = self.decoder(tgt, memory, tgt_is_causal=True, tgt_mask=self.tgt_mask)  # (B, P, C)
        action = self.output_layer(action)  # (B, P, 4)

        return action


class TransformerStudentSmall(nn.Module):
    def __init__(self, n_hist, n_pred):
        super().__init__()

        self.n_hist = n_hist
        self.n_pred = n_pred
        self.d_model = 256

        self.rgbd_tokenizer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
            nn.Sequential(*list(torchvision.models.efficientnet_b0().features)[1:]),
            nn.AvgPool2d(kernel_size=(7, 7))
        )

        self.proj_rgbd = nn.Linear(1280, 256)

        self.proj_state = nn.Linear(19, self.d_model)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True),
            num_layers=6
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8, batch_first=True),
            num_layers=6
        )

        self.readout = nn.Parameter(torch.zeros(1, self.n_pred, self.d_model))

        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        position_encoding = torch.zeros(2 * self.n_hist, self.d_model)
        position = torch.arange(0, 2 * self.n_hist, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, self.d_model, 2).float() * LOG10000 / self.d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encoding', position_encoding)

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(n_pred)
        self.register_buffer("tgt_mask", tgt_mask)
    
    def forward(self, x):

        # (B, H, 224, 224), (B, H, 3, 224, 224), (B, H, 6), (B, H, 15), (B, H, 4)
        depth, rgb, _, chaser_state, last_action = x

        # rgbd
        rgb = rgb.flatten(0, 1)  # (B*H, 3, 224, 224)
        depth = depth.flatten(0, 1).unsqueeze(1)  # (B*H, 1, 224, 224)
        rgbd = torch.concat([rgb, depth], dim=1)  # (B*H, 1, 224, 224)
        rgbd_token = self.rgbd_tokenizer(rgbd).squeeze()  # (B*H, 1280)
        rgbd_token = self.proj_rgbd(rgbd_token)  # (B*H, C)
        rgbd_token = rgbd_token.view(-1, self.n_hist, self.d_model)  # (B, H, C)
        
        # state
        state = torch.concat([chaser_state, last_action], dim=-1)  # (B, H, 19)
        state_token = self.proj_state(state)  # (B, H, C)

        src = torch.cat([
            rgbd_token,
            state_token
        ], dim=1)  # (B, 2*H, C)
        src = src + self.position_encoding
        
        memory = self.encoder(src)  # (B, 2*H, C)
        tgt = self.readout.expand(memory.shape[0], -1, -1)  # (B, P, C)
        action = self.decoder(tgt, memory, tgt_is_causal=True, tgt_mask=self.tgt_mask)  # (B, P, C)
        action = self.output_layer(action)  # (B, P, 4)
        
        return action
    

class TransformerVAEStudent(nn.Module):
    def __init__(self, n_hist, n_pred):
        super().__init__()
        self.n_hist = n_hist
        self.n_pred = n_pred
        self.d_model = 256
        self.rgbd_tokenizer = VAEEncoder(4)
        self.proj_state = nn.Linear(19, self.d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True),
            num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=self.d_model, nhead=8, batch_first=True),
            num_layers=6
        )
        self.readout = nn.Parameter(torch.zeros(1, self.n_pred, self.d_model))
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        # 方案一：空间和时间分别位置编码，非图像token空间位置编码设为0
        div_term = torch.exp(- torch.arange(0, 128, 2).float() * LOG10000 / 128.0)
        spatial_embedding_rgbd = torch.zeros(self.n_hist*196, 128)
        temporal_embedding_rgbd = torch.zeros(self.n_hist*196, 128)
        spatial_embedding_state = torch.zeros(self.n_hist, 128)
        temporal_embedding_state = torch.zeros(self.n_hist, 128)
        for i in range(n_hist):
            temporal_embedding_state[i, 0::2] = torch.sin(i * div_term)
            temporal_embedding_state[i, 1::2] = torch.cos(i * div_term)
            for j in range(196):
                k = i * 196 + j
                spatial_embedding_rgbd[k, 0::2] = torch.sin(j * div_term)
                spatial_embedding_rgbd[k, 1::2] = torch.cos(j * div_term)
                temporal_embedding_rgbd[k, 0::2] = torch.sin(i * div_term)
                temporal_embedding_rgbd[k, 1::2] = torch.cos(i * div_term)
        spatial_embedding = torch.concat([spatial_embedding_rgbd, spatial_embedding_state], dim=0)  # (H*197, 128)
        temporal_embedding = torch.concat([temporal_embedding_rgbd, temporal_embedding_state], dim=0)  # (H*197, 128)
        position_encoding = torch.concat([spatial_embedding, temporal_embedding], dim=-1)  # (H*197, 256)
        self.register_buffer('position_encoding', position_encoding)

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(n_pred)
        self.register_buffer("tgt_mask", tgt_mask)

    def forward(self, x):
        # (B, H, 224, 224), (B, H, 3, 224, 224), (B, H, 6), (B, H, 15), (B, H, 4)
        depth, rgb, _, chaser_state, last_action = x

        # rgbd
        depth = depth.flatten(0, 1)  # (B*H, 224, 224)
        depth = depth.unsqueeze(1)  # (B*H, 1, 224, 224)
        rgb = rgb.flatten(0, 1)  # (B*H, 3, 224, 224)
        rgbd = torch.concat([rgb, depth], dim=1)  # (B*H, 4, 224, 224)
        rgbd_tokens = self.rgbd_tokenizer(rgbd)  # (B*H, 512, 14, 14)
        rgbd_tokens = rgbd_tokens[:, :256, :, :]  # (B*H, 512, 14, 14)
        rgbd_tokens = rgbd_tokens.view(-1, self.n_hist, 256, 14, 14)  # (B, H, 256, 14, 14)
        rgbd_tokens = rgbd_tokens.permute(0, 1, 3, 4, 2).contiguous()  # (B, H, 14, 14, 256)'
        rgbd_tokens = rgbd_tokens.flatten(1, 3)  # (B, H*14*14, 256)

        # state
        state = torch.concat([chaser_state, last_action], dim=-1)  # (B, H, 19)
        state_tokens = self.proj_state(state)  # (B, H, 256)

        src = torch.concat([rgbd_tokens, state_tokens], dim=1)  # (B, H*197, 256)
        src = src + self.position_encoding  # (B, H*197, 256)
        memory = self.encoder(src)  # (B, H*197, 256)
        tgt = self.readout.expand(memory.shape[0], -1, -1)  # (B, P, 256)
        action = self.decoder(tgt, memory, tgt_is_causal=True, tgt_mask=self.tgt_mask)  # (B, P, 256)
        action = self.output_layer(action)  # (B, P, 4)
        
        return action


if __name__ == "__main__":
    
    batch_size = 16
    n_hist = 10
    n_pred = 5
