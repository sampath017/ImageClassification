import torch
from torch import nn


class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128, max_pool=True),
            ResBlock(128),

            ConvBlock(128, 256, max_pool=True),
            ConvBlock(256, 512, max_pool=True),
            ResBlock(512),
            nn.MaxPool2d(kernel_size=4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128, max_pool=True),
            ResBlock(128),

            ConvBlock(128, 256, max_pool=True),
            ConvBlock(256, 512, max_pool=True),
            ResBlock(512),

            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ResBlock(512),

            ConvBlock(512, 512),
            ConvBlock(512, 512),
            ResBlock(512),
            nn.MaxPool2d(kernel_size=4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        max_pool=False,
        last_block=False
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True if last_block else False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        ]

        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels=3,
        last_block=False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True if last_block else False
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x) + x

        return x


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        logits = torch.empty(10, dtype=torch.float)

        return logits
