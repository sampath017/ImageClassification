import lightning as L
import torch.nn.functional as F
import torch
from torch import nn
from utils import accuracy


class VGGBlock(nn.Module):
    def __init__(self,
                 block1_in_channels=3,
                 block1_out_channels=8,
                 block2_in_channels=8,
                 block2_out_channels=8,
                 ):
        super().__init__()
        self.block = nn.Sequential(
            # Block
            nn.Conv2d(
                in_channels=block1_in_channels,
                out_channels=block1_out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=block2_in_channels,
                out_channels=block2_out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

    def forward(self, x):
        return self.block(x)


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            VGGBlock(3, 8, 8, 8),
            VGGBlock(8, 8, 8, 8),
            VGGBlock(8, 8, 8, 8),
            VGGBlock(8, 8, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.Dropout(),
            nn.ReLU(),

            nn.Linear(16, 10)
        )

    def forward(self, x):
        features = self.feature_extractor(x).flatten(start_dim=1)
        logits = self.classifier(features)

        return logits


class VGGNetModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VGGNet()

    def forward(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        return loss, acc

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch):
        loss, acc = self.forward(batch)

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
