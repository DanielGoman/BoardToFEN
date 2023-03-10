from typing import Tuple

import torch
import torch.nn as nn


class PieceClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, num_type_classes: int,
                 num_color_classes: int):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=hidden_dim)
        self.conv_block2 = ConvBlock(in_channels=hidden_dim, out_channels=out_channels)

        self.linear = nn.Linear(in_features=16 * 8 * 8, out_features=8 * 8)

        self.type_linear = nn.Linear(in_features=8 * 8, out_features=num_type_classes)
        self.color_linear = nn.Linear(in_features=8 * 8, out_features=num_color_classes)

        self.relu = nn.ReLU()

        self.type_softmax = nn.LogSoftmax(dim=1)
        self.color_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.flatten(start_dim=1)
        x = self.linear(x)
        x = self.relu(x)

        type_scores = self.type_linear(x)
        color_scores = self.color_linear(x)

        type_probs = self.type_softmax(type_scores)
        color_probs = self.color_softmax(color_scores)

        return type_probs, color_probs


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x
