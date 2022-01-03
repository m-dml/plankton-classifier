import itertools
import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchmetrics
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score

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
        is_in_simclr_mode,
        batch_size,
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
        self.accuracy_func = torchmetrics.Accuracy()

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
        self.console_logger = utils.get_logger("LightningBaseModel", level=logging.INFO)
        self.is_in_simclr_mode = is_in_simclr_mode
        self.batch_size = batch_size

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
            global_batch_size = (
                self.trainer.num_nodes * self.trainer.gpus * self.batch_size
                if self.trainer.gpus > 0
                else self.batch_size
            )
            self.console_logger.info("global batch size is {}".format(global_batch_size))
            train_iters_per_epoch = len(self.trainer.datamodule.train_data) // global_batch_size
            self.console_logger.info(f"train iterations per epoch are {train_iters_per_epoch}")
            warmup_steps = train_iters_per_epoch * 10
            total_steps = train_iters_per_epoch * self.trainer.max_epochs
            if self.cfg_scheduler == "linear_warmup_decay":
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(
                        optimizer, linear_warmup_decay(warmup_steps, total_steps, cosine=True)
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "name": "Lars-LR",
                }
            elif self.cfg_scheduler == "cosine":
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=0.0),
                    "interval": "step",
                    "frequency": 1,
                    "name": "Cosine LR",
                }
            else:
                raise NotImplementedError(
                    f"The scheduler {self.cfg_scheduler} is not implemented. Please use one of "
                    f"[linear_warmup_decay, cosine] or none."
                )
            return [optimizer], [scheduler]

        return optimizer

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)
        features, classifier_outputs = self._do_gpu_parallel_step(images)

        return features, labels, classifier_outputs

    def training_step_end(self, training_step_outputs, *args, **kwargs):
        features, labels, classifier_outputs = training_step_outputs
        loss, acc = self._do_gpu_accumulated_step(classifier_outputs, labels, step="Training")
        return {"loss": loss, "features": features, "labels": labels, "classifier": classifier_outputs}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)
        self.console_logger.debug(f"Size of batch in validation_step: {len(labels)}")
        features, classifier_outputs = self._do_gpu_parallel_step(images)

        return features, labels, classifier_outputs

    def validation_step_end(self, validation_step_outputs, *args, **kwargs):
        features, labels, classifier_outputs = validation_step_outputs
        self.console_logger.debug(f"Size of batch in validation_step_end: {len(labels)}")
        loss, acc = self._do_gpu_accumulated_step(classifier_outputs, labels, step="Validation")
        return {"loss": loss, "features": features, "labels": labels, "classifier": classifier_outputs}

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)
        features, classifier_outputs = self._do_gpu_parallel_step(images)

        return features, labels, classifier_outputs

    def test_step_end(self, test_step_outputs, *args, **kwargs):
        features, labels, classifier_outputs = test_step_outputs
        loss, acc = self._do_gpu_accumulated_step(classifier_outputs, labels, step="Testing")
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

    def _do_gpu_parallel_step(self, images):
        features, classifier_outputs = self(images)
        return features, classifier_outputs

    def _do_gpu_accumulated_step(self, classifier_outputs, labels, step):
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

        return loss, accuracy

    def _update_accuracy_matrices(self, datagroup, labels_true, labels_est):
        # sum over batch to update confusion matrix
        n_classes = len(self.class_labels)
        idx = labels_true + n_classes * labels_est
        counts = torch.bincount(idx.reshape(-1), minlength=n_classes ** 2)
        self.confusion_matrix[datagroup] += counts.reshape((n_classes, n_classes))

    def _init_accuracy_matrices(self):
        n = len(self.class_labels)
        for datagroup in ["Validation", "Training", "Testing"]:
            self.confusion_matrix[datagroup] = torch.zeros(size=(n, n)).int()

    def _log_accuracy_matrices(self, datagroup):
        cm = self.confusion_matrix[datagroup]

        accuracy = torch.diag(cm).sum() / torch.maximum(torch.tensor(1.0), cm.sum())  # accuracy average over all data we're now logging
        conditional_probabilities = cm / torch.maximum(torch.tensor(1.0), cm.sum(axis=0))  # conditional probabilities for best guess
        conditional_accuracy = torch.diag(conditional_probabilities)
        for L, ca in zip(self.class_labels, conditional_accuracy):
            self.log(f"Cond. Acc/{L} {datagroup}", ca)

        self.plot_confusion_matrix(cm.cpu().numpy(), self.class_labels, f"Confusion_Matrix {datagroup}", title=f"Accuracy {accuracy}")
        self.plot_confusion_matrix(
            conditional_probabilities,
            self.class_labels,
            f"P(best guess | true) {datagroup}",
            title=f"P(best guess | true) {datagroup}",
        )

        # reset the CM
        self.confusion_matrix[datagroup] = torch.zeros((len(self.class_labels), len(self.class_labels))).int()

    def plot_confusion_matrix(self, cm, class_names, figname, title):
        figure = plt.figure(figsize=(15, 15))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = torch.arange(len(class_names))
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
        self._log_online_accuracy(features, labels)

    def _log_online_accuracy(self, x, y):
        if len(np.unique(y)) < 2:
            self.console_logger.warning("Not enough classes to evaluate classifier in online mode")
            return
        train_size = int(len(x) / 2)
        clf = SGDClassifier(max_iter=1000, tol=1e-3)
        clf.fit(x[:train_size], y[:train_size])
        predictions = clf.predict(x[train_size:])
        balanced_acc = balanced_accuracy_score(y_true=y[train_size:], y_pred=predictions)

        mean_acc = clf.score(x[train_size:], y[train_size:])
        self.log("Online Linear ACC", mean_acc)
        self.log("Online Linear balanced ACC", balanced_acc)

    def _create_tsne_image(self, labels, features, name):
        tsne = TSNE().fit_transform(features)

        tx, ty = tsne[:, 0], tsne[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        class_label_dict = dict()
        for i, label in enumerate(self.class_labels):
            class_label_dict[i] = label

        class_names = [class_label_dict[label] for label in labels.flatten().tolist()]
        df = pd.DataFrame({"x": tx.flatten(), "y": ty.flatten(), "label": class_names}).sort_values(by="label")
        num_points = len(df)
        fig, ax2 = plt.subplots(figsize=(8, 8))
        g = sns.jointplot(x="x", y="y", hue="label", data=df, ax=ax2, palette="deep", marginal_ticks=True, s=10)

        box = g.fig.axes[0].get_position()
        g.fig.axes[0].set_position([box.x0, box.y0 + box.height * 0.4, box.width, box.height * 0.6])

        box = g.fig.axes[2].get_position()
        g.fig.axes[2].set_position([box.x0, box.y0 + box.height * 0.4, box.width, box.height * 0.6])

        g.fig.axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
        g.fig.set_size_inches(8, 12)
        plt.suptitle(f"TSNE regression for {name} | {num_points} points")
        # plt.tight_layout()
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
