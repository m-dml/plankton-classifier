from pytorch_lightning import LightningModule
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser

from pytorch_lightning.metrics.functional import confusion_matrix


class PlanktonCLF(LightningModule):

    def __init__(self, num_classes, labels, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.feature_extractor = models.resnet18(
            pretrained=False,
            num_classes=num_classes)
        self.feature_extractor.eval()
        self.classifier = nn.Linear(in_features=num_classes, out_features=num_classes)
        self.labels = labels

    def setup(self, *args, **kwargs):
        self.logger.experiment.log_parameters(self.hparams)

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

        # logging using comet logger
        self.logger.experiment.log_metric("Loss/Train", avg_loss)

        self.logger.experiment.log_metric("Accuracy/Train", correct / total)

        epoch_dictionary = {
            # required
            'loss': avg_loss}

        return epoch_dictionary

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.nll_loss(y_hat, y)
        y_hat_classes = y_hat.argmax(dim=1)


        # identifying number of correct predections in a given batch
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        total = len(y)

        # logs- a dictionary
        logs = {"val_loss": val_loss}
        cm = confusion_matrix(target=y, pred=y_hat_classes, num_classes=self.num_classes)

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": val_loss,

            # optional for batch logging purposes
            "log": logs,

            # info to be used at epoch end
            "correct": correct,
            "total": total,
            "cm": cm.cpu().numpy()
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):

        cms = np.array([x['cm'] for x in outputs])
        cm = torch.sum(torch.from_numpy(cms), axis=0)

        self.logger.experiment.log_confusion_matrix(matrix=cm, labels=self.labels,
                                                    file_name=f"confusion_matrix_{self.current_epoch}.json")
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct = sum([x["correct"] for x in outputs])
        total = sum([x["total"] for x in outputs])

        # logging using comet logger
        self.logger.experiment.log_metric("Loss/Validation", avg_loss)
        self.logger.experiment.log_metric("Accuracy/Validation", correct / total)

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

        y_hat_classes = y_hat.argmax(dim=1)

        cm = confusion_matrix(target=y, pred=y_hat_classes, num_classes=self.num_classes)

        self.logger.experiment.log_metric("Loss/Test", loss)
        self.logger.experiment.log_metric("Accuracy/Test", correct / total)

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": loss,

            # info to be used at epoch end
            "correct": correct,
            "total": total,
            "cm": cm.cpu().numpy()
        }

        return batch_dictionary

    def test_epoch_end(self, outputs) -> None:

        cms = np.array([x['cm'] for x in outputs[:-1]])
        cm = torch.sum(torch.from_numpy(cms), axis=0)

        self.logger.experiment.log_confusion_matrix(matrix=cm, labels=self.labels, file_name=f"confusion_matrix_testing.json")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0002, help="adam: learning rate")

        return parser