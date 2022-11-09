import glob
import os
import pickle
import sys

import hydra
import torch
import torch.nn.functional as F
from natsort import natsorted
from pytorch_lightning import Trainer
from tqdm.auto import tqdm

sys.path.append("../..")
from src.models.LightningBaseModel import LightningModel  # noqa


def instantiate_model(ckpt_path, datamodule, example_input):
    model = LightningModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.set_external_data(
        class_labels=datamodule.unique_labels,
        all_labels=datamodule.all_labels,
        example_input_array=example_input.detach().cpu(),
    )
    return model


def instantiate_trainer(cfg):
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy=cfg.strategy,
        logger=[],
        callbacks=[],
        _convert_="partial",
        profiler=None,
    )
    return trainer


def return_if_file_exists(func):
    def function_wrapper(*args, **kwargs):
        return func

    file = function_wrapper()
    if os.path.isfile:
        return func
    else:
        raise FileNotFoundError(f"File {file} does not exist")


@return_if_file_exists
def get_temperature_file(checkpoint_file):
    if os.path.split(checkpoint_file)[-1] == "last.ckpt":
        path = os.path.split(checkpoint_file)[0]
        temp_files = natsorted(glob.glob(os.path.join(path, "temperatures_*.tensor")))
        return temp_files[-1]
    else:
        return checkpoint_file.replace("epoch=", "temperatures_").replace(".ckpt", ".tensor")


@return_if_file_exists
def get_distribution_file(checkpoint_file):
    path = os.path.split(checkpoint_file)[0]
    return os.path.join(path, "training_label_distribution.pt")


def run_and_save(
    _best_checkpoint,
    _dataloader,
    _return_metrics,
    _key,
    _experiment_number,
    data_splits_per_experiment,
    trainer,
    datamodule,
    example_input,
):
    model = instantiate_model(_best_checkpoint, datamodule, example_input)
    model.log_confusion_matrices = False
    model.temperature_scale = False
    _return_metrics[_key][_experiment_number] = trainer.test(model, _dataloader)[0]
    _return_metrics[_key][_experiment_number]["Data Fraction"] = data_splits_per_experiment[_experiment_number]
    _logits = torch.empty(size=(len(_dataloader.dataset), len(datamodule.unique_labels))).to("cuda:0")
    _labels = torch.empty(size=[len(_dataloader.dataset)]).to("cuda:0")

    with torch.no_grad():
        start = 0
        for i, batch in tqdm(enumerate(_dataloader), total=(len(_dataloader))):
            x, batch_labels = batch
            end = start + len(batch_labels[0])
            _labels[start:end] = batch_labels[0].squeeze()
            _logits[start:end, :] = model(x)[1]

            start = end

    _labels = _labels.detach().cpu().int()
    _logits = _logits.detach().cpu()

    torch.save(_logits, f"test_results/logits_{_key}_{_experiment_number}.pt")
    torch.save(_labels, f"test_results/labels_{_key}_{_experiment_number}.pt")
    with open(f"test_results/dict_{_key}_{_experiment_number}.pkl", "wb") as f:
        pickle.dump(_return_metrics, f)
    return _logits, _labels, _return_metrics


def get_confidence_and_acc_single(logits, labels, n_bins=20, logits_are_probs=False):
    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    if not logits_are_probs:
        softmaxes = F.softmax(logits, dim=1)
    else:
        softmaxes = logits
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    accuracy_bins = []
    confidence_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            accuracy_bins.append(accuracy_in_bin)
            confidence_bins.append(avg_confidence_in_bin)
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return torch.tensor(accuracy_bins), torch.tensor(confidence_bins), ece.item()


def get_best_checkpoints(path):
    best_checkpoints = []
    all_folders = glob.glob(os.path.join(path, "*"))
    experiment_folders = []
    for folder in all_folders:
        num_folder = os.path.split(folder)[-1]
        try:
            if num_folder.isnumeric():
                experiment_folders.append(folder)
        except:  # noqa E722
            print(f"Skipping {folder}")

    # fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 15), sharex=True, sharey=True)
    for experiment_number, experiment_path in enumerate(tqdm(experiment_folders)):
        # print(experiment_number)
        found_best_checkpoint = False
        with open(os.path.join(experiment_path, "main.log")) as f:
            complete_log = f.readlines()
            for line in complete_log:
                # print(line)
                if found_best_checkpoint:
                    best_checkpoint_part = os.path.normpath(line.strip().split("multirun/")[-1]).strip()
                    base_path = os.path.normpath(experiment_path.split("multirun")[0]).strip()
                    best_checkpoint_result = os.path.join(base_path, "multirun", best_checkpoint_part).strip()
                    # print(f"set best checkpoint to {best_checkpoint}")
                    break

                if "[main.main][INFO] - Best checkpoint path:" in line:
                    found_best_checkpoint = True
                    # print(f"found best checkpoint: {line}")

        if not found_best_checkpoint:
            print(f"Did not find checkpoint for {experiment_path}")
        else:
            best_checkpoints.append(best_checkpoint_result)

    return best_checkpoints
