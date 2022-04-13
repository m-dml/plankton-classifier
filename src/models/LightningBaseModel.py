import glob
import itertools
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning.utilities import rank_zero_only
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score

from src.external.temperature_scaling.temperature_scaling import ModelWithTemperature
from src.models.BaseModels import concat_feature_extractor_and_classifier
from src.utils import utils


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        log_images,
        log_confusion_matrices,
        log_tsne_image,
        optimizer,
        scheduler,
        feature_extractor,
        classifier,
        loss,
        metric,
        freeze_feature_extractor,
        is_in_simclr_mode,
        batch_size,
        temperature_scale,
        num_steps_per_epoch,
        num_unique_labels=None,
    ):

        super().__init__()

        self.training_class_counts = None
        self.num_steps_per_epoch = num_steps_per_epoch
        self.lr = optimizer.lr
        self.cfg_optimizer = optimizer
        self.cfg_loss = loss
        self.cfg_scheduler = scheduler
        self.cfg_metric = metric
        self.freeze_feature_extractor = freeze_feature_extractor
        self.save_hyperparameters(ignore=["example_input_array", "all_labels"])

        self.example_input_array = None
        self.class_labels = None
        self.all_labels = None
        self.loss_func = hydra.utils.instantiate(self.cfg_loss)
        self.accuracy_func = hydra.utils.instantiate(self.cfg_metric)

        self.feature_extractor: nn.Module = hydra.utils.instantiate(feature_extractor)
        if self.freeze_feature_extractor:
            self.feature_extractor.eval()
        # if num_classes in the config is set to None then use the number of classes found in the dataloader:
        self.classifier: nn.Module = hydra.utils.instantiate(
            classifier, num_classes=classifier.num_classes or num_unique_labels
        )
        self.model = concat_feature_extractor_and_classifier(
            feature_extractor=self.feature_extractor, classifier=self.classifier
        )
        self.temperatures = None

        self.log_images = log_images
        self.log_confusion_matrices = log_confusion_matrices
        self.log_tsne_image = log_tsne_image
        self.confusion_matrix = dict()
        self.console_logger = utils.get_logger("LightningBaseModel")
        self.console_logger.debug("Test Debug")
        self.is_in_simclr_mode = is_in_simclr_mode
        self.batch_size = batch_size
        self.temperature_scale = temperature_scale

    def set_external_data(self, class_labels, all_labels, example_input_array):

        self.example_input_array = example_input_array
        self.class_labels = class_labels
        self.all_labels = all_labels

        self._init_accuracy_matrices()

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

            # total_train_steps = len(self.trainer.train_dataloader)
            total_train_steps = int(
                (self.num_steps_per_epoch / (self.trainer.devices * self.trainer.num_nodes)) * self.trainer.max_epochs
            )
            self.console_logger.info("total_train_steps are {}".format(total_train_steps))

            if self.cfg_scheduler == "linear_warmup_decay":
                warmup_steps = int(total_train_steps * 0.01)  # Use 1% of training for warmup
                self.console_logger.info(
                    f"Total train steps are {total_train_steps}, so {warmup_steps} will be used for warmup."
                )
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(
                        optimizer, linear_warmup_decay(warmup_steps, total_train_steps, cosine=True)
                    ),
                    "interval": "step",
                    "frequency": 1,
                    "name": "Lars-LR",
                }
            elif self.cfg_scheduler == "cosine":
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_train_steps, eta_min=0.0),
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
        features, classifier_logits = self._do_gpu_parallel_step(images)

        return features, labels, label_names, classifier_logits

    def training_step_end(self, training_step_outputs, *args, **kwargs):
        features, labels, label_names, classifier_logits = training_step_outputs
        loss, acc = self._do_gpu_accumulated_step(classifier_logits, labels, label_names, step="Training")
        return {
            "loss": loss,
            "features": features.detach(),
            "labels": labels.detach(),
            "classifier": classifier_logits.detach(),
        }

    def on_train_epoch_end(self) -> None:
        self.console_logger.debug("Finished training epoch")

    def on_validation_epoch_start(self) -> None:
        self.console_logger.debug("Starting validation")

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)
        self.console_logger.debug(f"Size of batch in validation_step: {len(labels)}")
        features, classifier_logits = self._do_gpu_parallel_step(images)

        return features, labels, label_names, classifier_logits

    def validation_step_end(self, validation_step_outputs, *args, **kwargs):
        features, labels, label_names, classifier_logits = validation_step_outputs
        self.console_logger.debug(f"Size of batch in validation_step_end: {len(labels)}")
        loss, acc = self._do_gpu_accumulated_step(classifier_logits, labels, label_names, step="Validation")
        self.log("hp/loss", loss)
        self.log("hp/accuracy", acc)
        self.log("hp/epoch", torch.tensor(self.current_epoch).float())

        return {
            "loss": loss,
            "features": features.detach(),
            "labels": labels.detach(),
            "classifier": classifier_logits.detach(),
        }

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, labels, label_names = self._pre_process_batch(batch)
        features, classifier_logits = self._do_gpu_parallel_step(images)

        return features, labels, label_names, classifier_logits

    def test_step_end(self, test_step_outputs, *args, **kwargs):
        features, labels, label_names, classifier_logits = test_step_outputs
        loss, acc = self._do_gpu_accumulated_step(classifier_logits, labels, label_names, step="Testing")
        return {
            "loss": loss,
            "features": features.detach(),
            "labels": labels.detach(),
            "classifier": classifier_logits.detach(),
        }

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

    def _do_gpu_accumulated_step(self, classifier_outputs, labels, label_names, step):
        accuracy = 0  # set initial value, for the case of multi-label training
        predicted_labels = classifier_outputs.detach().argmax(dim=-1).unsqueeze(1)
        if isinstance(self.loss_func, torch.nn.KLDivLoss):
            loss = self.loss_func(F.log_softmax(classifier_outputs.float()), labels.float())
            accuracy = self.accuracy_func(predicted_labels, label_names, n_labels=classifier_outputs.size(1))
        else:
            targets = labels.detach().view(-1).to(torch.int).cpu()
            loss = self.loss_func(classifier_outputs, labels.view(-1).long())
            try:
                class_probabilities = F.softmax(classifier_outputs.detach(), dim=1).detach().cpu()
                accuracy = self.accuracy_func(class_probabilities, targets)
            except (RuntimeError, ValueError):
                pass

        # lets log some values for inspection (for example in tensorboard):
        self.log(f"loss/{step}", loss)
        self.log(f"Accuracy/{step}", accuracy)

        if self.log_confusion_matrices:
            self._update_accuracy_matrices(step, targets.cpu(), predicted_labels.cpu())

        return loss, accuracy

    def _update_accuracy_matrices(self, datagroup, labels_true, labels_est):
        # sum over batch to update confusion matrix
        n_classes = len(self.class_labels)
        idx = labels_true + n_classes * labels_est
        counts = torch.bincount(idx.reshape(-1), minlength=n_classes**2)
        self.confusion_matrix[datagroup] += counts.reshape((n_classes, n_classes))

    def _init_accuracy_matrices(self):
        n = len(self.class_labels)
        for datagroup in ["Validation", "Training", "Testing"]:
            self.confusion_matrix[datagroup] = torch.zeros(size=(n, n)).int()

    def _log_accuracy_matrices(self, datagroup):
        cm = self.confusion_matrix[datagroup]

        accuracy = torch.diag(cm).sum() / torch.maximum(
            torch.tensor(1.0), cm.sum()
        )  # accuracy average over all data we're now logging
        conditional_probabilities = cm / torch.maximum(
            torch.tensor(1.0), cm.sum(axis=0)
        )  # conditional probabilities for best guess
        conditional_accuracy = torch.diag(conditional_probabilities)
        for L, ca in zip(self.class_labels, conditional_accuracy):
            self.log(f"Cond. Acc/{L} {datagroup}", ca)

        self.plot_confusion_matrix(
            cm.cpu().numpy(), self.class_labels, f"Confusion_Matrix {datagroup}", title=f"Accuracy {accuracy}"
        )
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
        self.logger.experiment.add_figure(figname, figure, self.global_step)
        plt.close("all")
        return figure

    def validation_epoch_end(self, outputs):
        if self.log_confusion_matrices and self.current_epoch > 0:
            self._log_accuracy_matrices("Validation")
            # we also log and reset the training CM, so we log a training CM everytime we log a validation CM
            self._log_accuracy_matrices("Training")

        if self.log_tsne_image and self.current_epoch > 0:
            self.console_logger.debug("saving tsne image")
            self.plot_tsne_images(outputs)

        if self.temperature_scale:
            outputs_dict = {k: [dic[k] for dic in outputs] for k in outputs[0]}
            scaled_model = ModelWithTemperature(self, device=self.device)
            self.temperatures = scaled_model.get_temperature(
                labels=torch.cat(outputs_dict["labels"]).squeeze().long(),
                logits=torch.cat(outputs_dict["classifier"]),
                logger=self.console_logger,
            )

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

        g.fig.axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
        g.fig.set_size_inches(8, 12)
        plt.suptitle(f"TSNE regression for {name} | {num_points} points")
        # plt.tight_layout()
        self.logger.experiment.add_figure(f"TSNE Scatter {name}", g.fig, self.global_step)
        plt.close("all")

    def on_train_start(self):
        if self.trainer.datamodule.training_class_counts is not None and not self.is_in_simclr_mode:
            self.training_class_counts = torch.tensor(self.trainer.datamodule.training_class_counts).to(self.device)
            self.create_output_dir()

    @rank_zero_only
    def create_output_dir(self):
        save_path = self.trainer.checkpoint_callback.dirpath
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.training_class_counts, os.path.join(save_path, "training_label_distribution.pt"))

    def on_save_checkpoint(self, checkpoint) -> None:
        self.console_logger.debug("Running on_save_checkpoint")
        if self.automatic_optimization and (self.current_epoch == 0):
            return

        if self.is_in_simclr_mode:
            example_input = self.example_input_array[0]
        else:
            example_input = self.example_input_array

        # save model to onnx:
        folder = self.trainer.checkpoint_callback.dirpath
        onnx_file_generator = os.path.join(folder, f"model_{self.current_epoch}.onnx")

        self.console_logger.debug("exporting to onnx")
        torch.onnx.export(
            model=self.model.to(self.device),
            args=example_input.to(self.device),
            f=onnx_file_generator,
            opset_version=13,
            do_constant_folding=True,
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
        self.console_logger.debug("Saving state dict")
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(folder, f"complete_model_{self.current_epoch}.weights"))

        if self.temperature_scale:
            torch.save(self.temperatures, os.path.join(folder, f"temperatures_{self.current_epoch}.tensor"))

        best_epochs = self.get_best_epochs()
        self.remove_outdated_saves(best_epochs)

    @rank_zero_only
    def get_best_epochs(self):
        self.console_logger.debug("Getting best epochs")
        best_k_models = self.trainer.checkpoint_callback.best_k_models
        best_epochs = []
        for key, value in best_k_models.items():
            best_epochs.append(int(os.path.basename(key).replace("epoch=", "").replace(".ckpt", "")))

        self.console_logger.debug(f"Best inferred epochs are: {best_epochs}")
        self.console_logger.debug(f"Best models from callback are: {self.trainer.checkpoint_callback.best_k_models}")
        return best_epochs

    @rank_zero_only
    def remove_outdated_saves(
        self, best_epochs: list, filepatterns=("complete_model_*.weights", "model_*.onnx", "temperatures_*.tensor")
    ):
        self.console_logger.debug("Removing outdated files")
        folder = self.trainer.checkpoint_callback.dirpath
        for filepattern in filepatterns:
            files_to_keep = []
            for k in best_epochs:
                files_to_keep.append(os.path.join(folder, filepattern.replace("*", str(k))))

            # always keep the last epoch:
            files_to_keep.append(os.path.join(folder, filepattern.replace("*", str(self.current_epoch))))
            self.console_logger.debug(f"Files to keep: {files_to_keep}")
            files = glob.glob(os.path.join(folder, filepattern))
            files_to_delete = [item for item in files if item not in files_to_keep]
            self.console_logger.debug(f"All files found: {files}")
            for item in files_to_delete:
                self.console_logger.debug(f"Deleting: {item}")
                os.remove(item)
