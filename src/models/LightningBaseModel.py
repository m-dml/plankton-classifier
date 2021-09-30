import itertools
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from src.models.BaseModels import concat_feature_extractor_and_classifier
from src.utils import utils


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        class_labels,
        all_labels,
        example_input_array,
        log_images,
        log_confusion_matrices,
        log_tsne_image,
        optimizer,
        scheduler,
        feature_extractor,
        classifier,
        loss,
        freeze_feature_extractor,
        is_in_simclr_mode
    ):

        super().__init__()

        self.lr = optimizer.lr
        self.cfg_optimizer = optimizer
        self.cfg_loss = loss
        self.cfg_scheduler = scheduler
        self.freeze_feature_extractor = freeze_feature_extractor
        self.save_hyperparameters(ignore=["example_input_array", "all_labels"])

        self.example_input_array = example_input_array
        self.class_labels = class_labels
        self.all_labels = all_labels
        self.loss_func = hydra.utils.instantiate(self.cfg_loss)
        self.accuracy_func = pl_metrics.Accuracy()

        self.feature_extractor: nn.Module = hydra.utils.instantiate(feature_extractor)
        if self.freeze_feature_extractor:
            self.feature_extractor.eval()
        # if num_classes in the config is set to None then use the number of classes found in the dataloader:
        self.classifier: nn.Module = hydra.utils.instantiate(
            classifier, num_classes=classifier.num_classes or len(self.class_labels)
        )
        self.model = concat_feature_extractor_and_classifier(
            feature_extractor=self.feature_extractor, classifier=self.classifier
        )

        self.log_images = log_images
        self.log_confusion_matrices = log_confusion_matrices
        self.log_tsne_image = log_tsne_image
        self.confusion_matrix = dict()
        self._init_accuracy_matrices()
        self.console_logger = utils.get_logger("LightningBaseModel")
        self.is_in_simclr_mode = is_in_simclr_mode

    # def setup(self, *args, **kwargs):
    #     self.model = self.model.to(self.device)
    #     self.console_logger.info("Logging model graph")
    #     self.logger.experiment[0].add_graph(self.to("cpu"), self.example_input_array.to("cpu"))
    #     self.console_logger.info("Successful saved model graph")

    def forward(self, images, *args, **kwargs):
        if self.is_in_simclr_mode:
            features_0 = self.feature_extractor(images[0])
            features_1 = self.feature_extractor(images[1])
            class_log_probabilities_0 = self.classifier(features_0)
            class_log_probabilities_1 = self.classifier(features_1)
            predictions = torch.cat([class_log_probabilities_0, class_log_probabilities_1], dim=0)
            features = torch.cat([features_0, features_1], dim=0)
        else:
            features = self.feature_extractor(images)
            predictions = self.classifier(features)
        return features, predictions

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg_optimizer, params=self.model.parameters(), lr=self.lr)
        if self.cfg_scheduler:
            scheduler = hydra.utils.instantiate(self.cfg_scheduler, optimizer=optimizer)
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)
        log_images = self.log_images and batch_idx == 0
        loss, acc, classifier_outputs, features = self._do_step(images, labels, label_names, step="Training", log_images=log_images)
        return {"loss": loss, "features": features, "labels": labels, "classifier": classifier_outputs}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)

        log_images = self.log_images and batch_idx == 0
        loss, acc, classifier_outputs, features = self._do_step(images, labels, label_names, step="Validation", log_images=log_images)
        return {"loss": loss, "features": features, "labels": labels, "classifier": classifier_outputs}

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)

        loss, acc, classifier_outputs, features = self._do_step(images, labels, label_names, step="Testing", log_images=False)
        return {"loss": loss, "features": features, "labels": labels, "classifier": classifier_outputs}

    def _pre_process_batch(self, batch):
        images, labels = batch
        if len(labels) == 2:
            labels, label_names = labels
        else:
            label_names = torch.tensor([0 for _ in range(len(labels))])

        if isinstance(images, (tuple, list)):
            labels = torch.cat([labels, labels], dim=0)

        return images, labels, label_names

    def _do_step(self, images, labels, label_names, step, log_images=False):

        features, classifier_outputs = self(images)
        class_probabilities = F.softmax(classifier_outputs.detach(), dim=1).detach().cpu()
        labels_est = classifier_outputs.detach().argmax(dim=-1).cpu()
        targets = labels.detach().view(-1).to(torch.int).cpu()

        loss = self.loss_func(classifier_outputs, labels.view(-1).long())
        try:
            accuracy = self.accuracy_func(class_probabilities, targets)
        except (RuntimeError, ValueError):
            accuracy = 0

        # lets log some values for inspection (for example in tensorboard):
        self.log(f"loss/{step}", loss)
        self.log(f"Accuracy/{step}", accuracy)

        if self.log_confusion_matrices:
            self._update_accuracy_matrices(step, targets, labels_est)

        if log_images:
            self.console_logger.warning("Logging of images is deprecated.")

        return loss, accuracy, classifier_outputs.detach(), features.detach()

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
            self.log(f"Cond. Acc/{L} {datagroup}", ca)

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

    def validation_epoch_end(self, outputs):
        if self.log_confusion_matrices:
            self._log_accuracy_matrices("Validation")
            # we also log and reset the training CM, so we log a training CM everytime we log a validation CM
            self._log_accuracy_matrices("Training")

        if self.log_tsne_image:
            self.plot_tsne_images(outputs)

    def test_epoch_end(self, outputs):
        if self.log_confusion_matrices:
            self._log_accuracy_matrices("Testing")

        if self.log_tsne_image:
            self.plot_tsne_images(outputs)

    def plot_tsne_images(self, outputs):
        if isinstance(outputs[0], list):
            outputs = [item for sublist in outputs for item in sublist]

        res = dict()
        for key in outputs[0]:
            res[key] = []
            for ele in outputs:
                res[key].append(ele[key])

        labels = torch.cat(res["labels"], dim=0).detach().cpu().numpy()
        features = torch.cat(res["features"], dim=0).detach().cpu().numpy()
        classifier_outputs = torch.cat(res["classifier"], dim=0).detach().cpu().numpy()
        self._create_tsne_image(labels, classifier_outputs, "Classifier")
        self._create_tsne_image(labels, features, "Feature Extractor")

    def _create_tsne_image(self, labels, features, name):
        tsne = TSNE().fit_transform(features)

        tx, ty = tsne[:, 0], tsne[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        class_label_dict = dict()
        for i, label in enumerate(self.class_labels):
            class_label_dict[i] = label

        class_names = [class_label_dict[label] for label in labels]
        df = pd.DataFrame({"x": tx.flatten(), "y": ty.flatten(), "label": class_names}).sort_values(by="label")
        num_points = len(df)
        fig, ax2 = plt.subplots(figsize=(8, 8))
        g = sns.jointplot(x="x", y="y", hue="label", data=df, ax=ax2, palette="deep", marginal_ticks=True, s=10)
        plt.suptitle(f"TSNE regression for {name} | {num_points} points")
        plt.tight_layout()
        self.logger.experiment[0].add_figure(f"TSNE Scatter {name}", g.fig, self.global_step)
        plt.close("all")

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.automatic_optimization and (self.current_epoch == 0):
            return

        # save model to onnx:
        folder = self.trainer.checkpoint_callback.dirpath
        onnx_file_generator = os.path.join(folder, f"complete_model_{self.global_step}.onnx")
        if self.is_in_simclr_mode:
            example_input = self.example_input_array[0]
        else:
            example_input = self.example_input_array
        torch.onnx.export(
            model=self.model,
            args=example_input.to(self.device),
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

        # save the feature_extractor_weights:
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(folder, f"complete_model_{self.global_step}.weights"))
