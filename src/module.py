import torch.nn.functional as F
import torch
from utils import accuracy
from models import ToyNet, ResNet18, ResNet9, ResNet56
from temp import resnet18


class QuickModule:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

    def forward(self, batch):
        self.model = self.model.to(self.device)
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        return loss, acc


class ResNetModule(QuickModule):
    def __init__(self, model=None, toy_model=False):
        super().__init__()
        if model:
            self.model = model
        elif toy_model:
            self.model = ToyNet()
        else:
            self.model = ResNet18(num_classes=100)

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def validation_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc
