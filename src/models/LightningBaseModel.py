import itertools
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as base_model


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        class_labels,
        all_labels,
        example_input_array,
        log_images,
        log_confusion_matrices,
        use_weighted_loss,
        optimizer,
        model,
    ):

        super().__init__()

        self.cfg_optimizer = optimizer
        self.save_hyperparameters()

        self.example_input_array = example_input_array
        self.class_labels = class_labels
        self.all_labels = all_labels
        self.label_weight_tensor = self.get_label_weights()
        if use_weighted_loss:
            self.loss_func = nn.NLLLoss(weight=self.label_weight_tensor)
        else:
            self.loss_func = nn.NLLLoss()
        self.accuracy_func = pl_metrics.Accuracy()
        self.model = hydra.utils.instantiate(model)
        self.log_images = log_images
        self.log_confusion_matrices = log_confusion_matrices
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

    def forward(self, images, *args, **kwargs):
        predictions = self.model(images)
        return predictions

    def configure_optimizers(self):
        self.console_logger.info(f"Instantiating optimizer <{self.cfg_optimizer._target_}>")
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=self.parameters())
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
        for datagroup in ["Validation", "Training", "Testing"]:
            self.confusion_matrix[datagroup] = np.zeros((n, n), dtype=np.int64)

    def _log_accuracy_matrices(self, datagroup):
        cm = self.confusion_matrix[datagroup]

        accuracy = np.diag(cm).sum() / np.maximum(1.0, cm.sum())  # accuracy average over all data we're now logging
        conditional_probabilities = cm / np.maximum(1.0, cm.sum(axis=0))  # conditional probabilities for best guess
        conditional_accuracy = np.diag(conditional_probabilities)
        for L, ca in zip(self.class_labels, conditional_accuracy):
            self.log(f"Cond. Acc. {L} {datagroup}", ca)

        self.plot_confusion_matrix(cm, self.class_labels, f"Confusion_Matrix {datagroup}", title=f"Accuracy {accuracy}")
        self.plot_confusion_matrix(
            conditional_probabilities,
            self.class_labels,
            f"P(best guess | true) {datagroup}",
            title=f"P(best guess | true) {datagroup}",
        )

        # reset the CM
        self.confusion_matrix[datagroup] = np.zeros((len(self.class_labels), len(self.class_labels)), dtype=np.int64)

    def plot_confusion_matrix(self, cm, class_names, figname, title):
        figure = plt.figure(figsize=(15, 15))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.0

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            if cm[i, j] > 0.01:
                if int(cm[i, j]) == float(cm[i, j]):
                    plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color=color, fontsize=8)
                else:
                    plt.text(j, i, "{:.2f}".format(cm[i, j]), horizontalalignment="center", color=color, fontsize=8)

        plt.tight_layout()
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        self.logger.experiment[0].add_figure(figname, figure, self.global_step)
        plt.close("all")
        return figure

    def on_validation_epoch_end(self):
        self._log_accuracy_matrices("Validation")
        # we also log and reset the training CM, so we log a training CM everytime we log a validation CM
        self._log_accuracy_matrices("Training")

    def on_test_epoch_end(self):
        self._log_accuracy_matrices("Testing")

    def on_save_checkpoint(self, checkpoint) -> None:
        # save model to onnx:
        folder = self.trainer.checkpoint_callback.dirpath
        onnx_file_generator = os.path.join(folder, f"model_{self.global_step}.onnx")
        torch.onnx.export(
            model=self.model,
            args=self.example_input_array.to(self.device),
            f=onnx_file_generator,
            opset_version=12,
            verbose=False,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # makes the batch-size variable for inference
                "output": {0: "batch_size"},
            },
        )
