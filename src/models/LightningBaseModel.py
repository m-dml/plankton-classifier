import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18


class LightningModel(pl.LightningModule):

    def __init__(self, class_labels, *args, **kwargs):

        super().__init__()
        self.class_labels = class_labels
        self.model = self.define_model()
        self.learning_rate = kwargs["learning_rate"]
        self.save_hyperparameters()

    def define_model(self):
        return resnet18(pretrained=False, num_classes=len(self.class_labels))

    def forward(self, images,  *args, **kwargs):
        predictions = self.model(images)
        return F.log_softmax(predictions, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)
        logging.debug(f"labels have the shape: {labels.shape}")
        logging.debug(f"predictions have the shape: {predictions.shape}")

        loss_func = nn.NLLLoss()
        loss = loss_func(predictions, labels.view(-1).long())

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Training", loss)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)

        loss_func = nn.NLLLoss()
        loss = loss_func(predictions, labels.view(-1).long())

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Validation", loss)

        if batch_idx == 0:
            self.log_confusion_matrix(predictions, labels)

        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        # for the toy example we will not use the meta_data and only the images to make a prediction.
        predictions = self(images)

        loss_func = nn.NLLLoss()
        loss = loss_func(predictions, labels.view(-1).long())

        # lets log some values for inspection (for example in tensorboard):
        self.log("NLL Test", loss)

        return loss

    def log_confusion_matrix(self, predictions, targets):
        conf_mat = confusion_matrix(torch.argmax(predictions, dim=-1).detach().cpu(), targets.detach().cpu(),
                                    labels=np.arange(len(self.class_labels)))

        fig, ax = plt.subplots()
        ax.imshow(conf_mat)
        ax.set_xticklabels(self.class_labels)
        ax.set_yticklabels(self.class_labels)
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        self.logger.experiment[0].add_figure("Confusion_Matrix", fig, self.global_step)
        plt.close("all")
