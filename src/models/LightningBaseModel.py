import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18
import pytorch_lightning.metrics as pl_metrics


class LightningModel(pl.LightningModule):

    def __init__(self, class_labels, *args, **kwargs):

        super().__init__()
        self.class_labels = class_labels
        self.model = self.define_model()
        self.learning_rate = kwargs["learning_rate"]
        self.loss_func = nn.NLLLoss()
        self.accuracy_func = pl_metrics.Accuracy()
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

        loss, acc = self._do_step(images, labels, label_names, step="Validation", log_images=False)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        log_images = False
        if batch_idx == 0:
            log_images = True

        loss, acc = self._do_step(images, labels, label_names, step="Validation", log_images=log_images)
        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        loss, acc = self._do_step(images, labels, label_names, step="Validation", log_images=False)
        return loss

    def _do_step(self, images, labels, label_names, step, log_images=False):

        predictions = self(images)
        loss = self.loss_func(predictions, labels.view(-1).long())
        accuracy = self.accuracy_func(F.softmax(predictions, dim=1).detach().cpu(), labels.to(torch.int).detach().cpu())

        # lets log some values for inspection (for example in tensorboard):
        self.log(f"NLL {step}", loss)
        self.log(f"Accuracy {step}", accuracy)

        if log_images:
            self.log_confusion_matrix(predictions, labels)
            self.log_images(images, label_names)

        return loss, accuracy

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

    def log_images(self, images, labels):
        if self.hparams.batch_size >= 16:
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
            for i in range(16):
                ax = axes.flatten()[i]
                ax.imshow(images[i].detach().cpu().moveaxis(0, -1))
                ax.set_title(labels[i])

            self.logger.experiment[0].add_figure("Image Matrix", fig, self.global_step)
            plt.close("all")
