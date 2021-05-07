import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class LightningModel(pl.LightningModule):

    def __init__(self, class_labels, all_labels, example_input_array, *args, **kwargs):

        super().__init__()
        self.example_input_array = example_input_array
        self.class_labels = class_labels
        self.all_labels = all_labels
        self.label_weight_tensor = self.get_label_weights()
        self.model = self.define_model(pretrained=kwargs['use_pretrained'])
        self.learning_rate = kwargs["learning_rate"]
        if kwargs["use_weighted_loss"]:
            self.loss_func = nn.NLLLoss(weight=self.label_weight_tensor)
        else:
            self.loss_func = nn.NLLLoss()
        self.accuracy_func = pl_metrics.Accuracy()
        self.save_hyperparameters()
        self.log_images = kwargs['log_images']
        self.log_confusion_matrices = kwargs['log_confusion_matrices']
        self.confusion_matrix = dict()
        self._init_accuracy_matrices()

    def get_label_weights(self):
        label_weights_dict = dict()

        for label in self.all_labels:
            if label in label_weights_dict.keys():
                label_weights_dict[label] += 1
            else:
                label_weights_dict[label] = 1

        weights = []
        for label in self.class_labels:
            weights.append(1 / label_weights_dict[label])

        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        return weight_tensor

    def define_model(self, input_channels=3, pretrained=False):
        feature_extractor = resnet50(pretrained=pretrained, num_classes=1000)
        feature_extractor.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        classifier = nn.Linear(1000, len(self.class_labels))

        model = nn.Sequential(feature_extractor, classifier)
        return model

    def forward(self, images,  *args, **kwargs):
        predictions = self.model(images)
        return F.log_softmax(predictions, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        log_images = self.log_images and batch_idx == 0

        loss, acc = self._do_step(images, labels, label_names, step="Training", log_images=log_images)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        log_images = self.log_images and batch_idx == 0

        loss, acc = self._do_step(images, labels, label_names, step="Validation", log_images=log_images)
        return loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = batch

        loss, acc = self._do_step(images, labels, label_names, step="Testing", log_images=False)
        return loss

    def _do_step(self, images, labels, label_names, step, log_images=False):

        class_log_probabilities = self(images)
        class_probabilities = F.softmax(class_log_probabilities, dim=1).detach().cpu()
        labels_est = class_log_probabilities.argmax(dim=-1).detach().cpu()
        targets = labels.view(-1).to(torch.int).detach().cpu()

        loss = self.loss_func(class_log_probabilities, labels.view(-1).long())
        accuracy = self.accuracy_func(class_probabilities, targets)

        # lets log some values for inspection (for example in tensorboard):
        self.log(f"NLL {step}", loss)
        self.log(f"Accuracy {step}", accuracy)

        if self.log_confusion_matrices:
            self._update_accuracy_matrices(step, targets, labels_est)

        if log_images:
            self.log_images(images, label_names)

        return loss, accuracy

    def _update_accuracy_matrices(self, datagroup, labels_true, labels_est):
        # sum over batch to update confusion matrix
        n_classes = len(self.class_labels)
        idx = labels_true + n_classes * labels_est
        counts = np.bincount(idx.reshape(-1), minlength=n_classes ** 2)
        self.confusion_matrix[datagroup] += counts.reshape((n_classes, n_classes))

    def _init_accuracy_matrices(self):
        n = len(self.class_labels)
        for datagroup in ['Validation', 'Training', 'Testing']:
            self.confusion_matrix[datagroup] = np.zeros((n, n), dtype=np.int64)

    def _log_accuracy_matrices(self, datagroup):
        cm = self.confusion_matrix[datagroup]

        accuracy = np.diag(cm).sum() / np.maximum(1.0, cm.sum())  # accuracy average over all data we're now logging
        conditional_probabilities = cm / np.maximum(1.0, cm.sum(axis=0))  # conditional probabilities for best guess
        conditional_accuracy = np.diag(conditional_probabilities)
        for L, ca in zip(self.class_labels, conditional_accuracy):
            self.log(f"Cond. Acc. {L} {datagroup}", ca)

        self._log_img(cm, f"Confusion_Matrix {datagroup}",
                      ticklabels=self.class_labels, xlabel='Target', ylabel='Prediction', title=f'Accuracy {accuracy}')
        self._log_img(conditional_probabilities, f"P(best guess | true) {datagroup}",
                      ticklabels=self.class_labels, xlabel='Target', ylabel='Prediction')

        # reset the CM
        self.confusion_matrix[datagroup] = np.zeros((len(self.class_labels), len(self.class_labels)), dtype=np.int64)

    def _log_img(self, x, figname, show_colorbar=True, ticklabels=None, xlabel=None, ylabel=None, title=None):
        h, w = x.shape[0:2]
        x[np.isnan(x)] = 0.0
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(x)
        if show_colorbar:
            fig.colorbar(mappable=im, ticks=[0, x.max()])
        if ticklabels is not None and len(ticklabels) == w:
            plt.xticks(np.arange(w), ticklabels, rotation='vertical')
        if ticklabels is not None and len(ticklabels) == h:
            plt.yticks(np.arange(h), self.class_labels)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.axis('tight')
        self.logger.experiment[0].add_figure(figname, fig, self.global_step)
        plt.close('all')

    def on_validation_epoch_end(self):
        self._log_accuracy_matrices('Validation')
        # we also log and reset the training CM, so we log a training CM everytime we log a validation CM
        self._log_accuracy_matrices('Training')

    def on_test_epoch_end(self):
        self._log_accuracy_matrices('Testing')

    def log_images(self, images, labels):
        if self.hparams.batch_size >= 16:
            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
            for i in range(16):
                ax = axes.flatten()[i]
                ax.imshow(images[i].detach().cpu().moveaxis(0, -1))
                ax.set_title(labels[i])

            self.logger.experiment[0].add_figure("Image Matrix", fig, self.global_step)
            plt.close("all")

    def on_save_checkpoint(self, checkpoint) -> None:
        # save model to onnx:
        folder = self.trainer.checkpoint_callback.dirpath
        onnx_file_generator = os.path.join(folder, f"model_{self.global_step}.onnx")
        torch.onnx.export(model=self.model,
                          args=self.example_input_array.to(self.device),
                          f=onnx_file_generator,
                          opset_version=12,
                          verbose=False,
                          export_params=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},  # makes the batch-size variable for inference
                                        'output': {0: 'batch_size'}}
                          )
