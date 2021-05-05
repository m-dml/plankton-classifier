import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18


class LightningModel(pl.LightningModule):

    def __init__(self, class_labels, all_labels, example_input_array, *args, **kwargs):

        super().__init__()
        self.example_input_array = example_input_array
        self.class_labels = class_labels
        self.all_labels = all_labels
        self.label_weight_tensor = self.get_label_weights()
        self.model = self.define_model()
        self.learning_rate = kwargs["learning_rate"]
        if kwargs["use_weighted_loss"]:
            self.loss_func = nn.NLLLoss(weight=self.label_weight_tensor)
        else:
            self.loss_func = nn.NLLLoss()
        self.accuracy_func = pl_metrics.Accuracy()
        self.save_hyperparameters()

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

        loss, acc = self._do_step(images, labels, label_names, step="Training", log_images=False)
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

        loss, acc = self._do_step(images, labels, label_names, step="Test", log_images=False)
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