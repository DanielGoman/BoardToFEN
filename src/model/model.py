import numpy as np
import torch
import torch.nn as nn


class PieceClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int, num_classes: int):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=hidden_dim)
        self.conv_block2 = ConvBlock(in_channels=hidden_dim, out_channels=out_channels)
        self.conv_block3 = ConvBlock(in_channels=out_channels, out_channels=out_channels * 4)

        self.linear = nn.Linear(in_features=(16 * 4) * 8 * 8, out_features=4 * 4 * 4)

        self.class_linear = nn.Linear(in_features=4 * 4 * 4, out_features=num_classes)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on the received tensor

        Args:
            x: tensor of shape (Batch, Channels, Height, Width)

        Returns:
            class_scores: class scores of shape (Batch, N classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.flatten(start_dim=1)
        x = self.linear(x)
        x = self.relu(x)

        class_scores = self.class_linear(x)

        return class_scores

    def inference(self, x: torch.Tensor) -> np.ndarray:
        """Performs a froward pass and calculates the predicted classes for every item in the batch

        Args:
            x: tensor of shape (Batch, Channels, Height, Width)

        Returns:
            class_preds: predicted classes per item in the batch, of shape (Batch, 1)

        """
        class_scores = self.forward(x)
        class_preds = torch.argmax(torch.softmax(class_scores, dim=1), dim=1).cpu()

        return class_preds


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
