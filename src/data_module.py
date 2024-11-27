from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import torch
import lightning as L


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, path, batch_size=64):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.transform = v2.Compose([
            # Normalize
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
        ])

    def prepare_data(self):
        dataset = CIFAR10(self.path, train=True, download=True)

    def setup(self, stage):
        if stage == "fit":
            self.dataset = CIFAR10(self.path, train=True,
                                   transform=self.transform)
            train_size = int(len(self.dataset) * 0.7)
            val_size = len(self.dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
