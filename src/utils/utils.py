import logging
import os
import pickle
import warnings

import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from src.utils import LOG_LEVEL


def get_logger(name=__name__, level=None) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    new_logger = logging.getLogger(name)
    if not level:
        level_obj = logging.getLevelName(LOG_LEVEL.log_level.strip().upper())
    else:
        level_obj = logging.getLevelName(level.strip().upper())
    new_logger.setLevel(level_obj)

    return new_logger


def set_log_levels(level="INFO"):
    LOG_LEVEL.log_level = level.strip().upper()
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.getLevelName("INFO"))
        for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
            setattr(logger, level, rank_zero_only(getattr(logger, level)))


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True
        log.setLevel(logging.DEBUG)

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.callbacks.get("gpu_monitoring"):
            config.callbacks.gpu_monitoring = [None]

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>")
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
) -> dict:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["lightning_module"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    return hparams


def eval_and_save(checkpoint_file, trainer, datamodule, example_input):
    from src.models.LightningBaseModel import LightningModel

    def instantiate_model(ckpt_path, _datamodule, _example_input):
        _model = LightningModel.load_from_checkpoint(checkpoint_path=ckpt_path)
        _model.set_external_data(
            class_labels=_datamodule.unique_labels,
            all_labels=_datamodule.all_labels,
            example_input_array=_example_input.detach().cpu(),
        )
        return _model

    def infer_key_and_experiment_and_epoch_from_file(_checkpoint_file):
        path = os.path.normpath(_checkpoint_file)
        path_list = path.split(os.sep)

        _key = None
        _experiment = None
        _epoch = None

        for i, element in enumerate(path_list):
            if element == "plankton_logs":
                _experiment = path_list[i+1] + "_singlelabel"

            elif element == "logs":
                _key = path_list[i-1]

            elif "epoch=" in element:
                _epoch = element.replace("epoch=", "").replace(".ckpt", "")

        if any([return_component is None for return_component in [_key, _experiment, _epoch]]):
            raise ValueError("key, experiment or epoch is not set!")

        return int(_key), _experiment, _epoch

    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    data_splits_per_experiment = [np.round(x, 2) for x in np.arange(0.01, 0.1, 0.01)] + [np.round(x, 2) for x in
                                                                                         np.arange(0.1, 1.1, 0.1)]
    key, experiment, epoch = infer_key_and_experiment_and_epoch_from_file(checkpoint_file)
    return_metrics = dict()
    return_metrics[experiment] = dict()
    model = instantiate_model(checkpoint_file, datamodule, example_input)
    model.log_confusion_matrices = False
    model.temperature_scale = False
    return_metrics[experiment][key] = trainer.test(model, dataloader)[0]
    return_metrics[experiment][key]["Data Fraction"] = data_splits_per_experiment[key]
    return_metrics[experiment][key]["Best Epoch"] = epoch

    logits = torch.empty(size=(len(dataloader.dataset), len(datamodule.unique_labels))).to("cuda:0")
    labels = torch.empty(size=[len(dataloader.dataset)]).to("cuda:0")

    with torch.no_grad():
        start = 0
        for i, batch in enumerate(dataloader):
            x, batch_labels = batch
            end = start + len(batch_labels[0])
            labels[start: end] = batch_labels[0].squeeze()
            logits[start: end, :] = model(x)[1]
            start = end

    labels = labels.detach().cpu().int()
    logits = logits.detach().cpu()

    if not os.path.isdir("test_results"):
        os.makedirs("test_results")
    torch.save(logits, f"test_results/logits_{experiment}_{key}.pt")
    torch.save(labels, f"test_results/labels_{experiment}_{key}.pt")
    with open(f"test_results/dict_{experiment}_{key}.pkl", 'wb') as f:
        pickle.dump(return_metrics, f)
    return logits, labels, return_metrics
