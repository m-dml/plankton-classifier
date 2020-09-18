from pytorch_lightning import LightningModule
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F


class LitModel(LightningModule):

    def __init__(self):
        super().__init__()
        num_target_classes = 5
        self.feature_extractor = models.resnet18(
            pretrained=False,
            num_classes=num_target_classes)
        self.feature_extractor.eval()
        self.classifier = nn.Linear(in_features=num_target_classes, out_features=num_target_classes)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.nll_loss(y_hat, y)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        return loss
