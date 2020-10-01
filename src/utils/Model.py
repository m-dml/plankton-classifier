from pytorch_lightning import LightningModule
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


class PlanktonCLF(LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
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
        lr = self.hparams.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)

        # identifying number of correct predections in a given batch
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        # logs- a dictionary
        logs = {"train_loss": loss}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,

            # optional for batch logging purposes
            "log": logs,

            # info to be used at epoch end
            "correct": correct,
            "total": total
        }

        return batch_dictionary

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        if self.current_epoch == 1:
            sample_img = torch.rand((1, 3, 500, 500))
            # self.logger.experiment.add_graph(PlanktonCLF(), sample_img)

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          correct / total,
                                          self.current_epoch)

        epoch_dictionary = {
            # required
            'loss': avg_loss}

        return epoch_dictionary

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.nll_loss(y_hat, y)

        # identifying number of correct predections in a given batch
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        # logs- a dictionary
        logs = {"val_loss": val_loss}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": val_loss,

            # optional for batch logging purposes
            "log": logs,

            # info to be used at epoch end
            "correct": correct,
            "total": total
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          correct / total,
                                          self.current_epoch)

        epoch_dictionary = {
            # required
            'loss': avg_loss}

        return epoch_dictionary

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        total = len(y)
        correct = y_hat.argmax(dim=1).eq(y).sum().item()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test",
                                          loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Test",
                                          correct / total,
                                          self.current_epoch)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0002, help="adam: learning rate")

        return parser
