import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResidualBlock(nn.Module):
    def __init__(self, channels, downsample = None):
        super().__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)

# This is an extractor specifically used with SL training
class ResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: spaces.Box, 
                 features_dim: int = 256,
                 hidden_dim=64, 
                 num_blocks=4
                 ):
        super().__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_obs = torch.as_tensor(observation_space.sample()[None]).float()
            out = self.conv1(dummy_obs)
            out = self.res_blocks(out)
            out = self.flatten(out)
            n_flatten = out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.flatten(out)
        out = self.linear(out)

        return out