import os
import platform
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from torchvision.transforms import Compose

from src.lib.config import Config, register_configs
from src.utils import utils

# sometimes windows and matplotlib don't play well together. Therefore we have to configure win for plt:
if platform.system() == "Windows":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# register the structured configs:
register_configs()

# set up advanced logging:
log = utils.get_logger(__name__)
pl_logger = utils.get_logger("pytorch_lightning")
pl_logger.handlers = []
pl_logger.addHandler(log)


@hydra.main(config_name="config", config_path="conf")
def main(cfg: Config):

    utils.extras(cfg)

    # Pretty print config using Rich library
    if cfg.print_config:
        utils.print_config(cfg, resolve=True)

    torch.manual_seed(cfg.random_seed)
    pl.seed_everything(cfg.random_seed)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Transformations
    train_transforms: Compose = hydra.utils.instantiate(cfg.datamodule.train_transforms)
    valid_transforms: Compose = hydra.utils.instantiate(cfg.datamodule.valid_transforms)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule,
        train_transforms=train_transforms,
        valid_transforms=valid_transforms,
        dataset=cfg.datamodule.dataset,
    )
    datamodule.setup()

    # generate example input array:
    for batch in datamodule.train_dataloader():
        example_input, _ = batch
        if isinstance(example_input, tuple):
            example_input = torch.stack(example_input).detach().cpu()
        break

    # Init Lightning model
    log.info(f"Instantiating model <{cfg.lightning_module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.lightning_module,
        optimizer=cfg.optimizer,
        feature_extractor=cfg.model.feature_extractor,
        classifier=cfg.model.classifier,
        loss=cfg.loss,
        class_labels=datamodule.unique_labels,
        all_labels=datamodule.all_labels,
        example_input_array=example_input.detach().cpu(),
    )

    if cfg.load_state_dict is not None:
        model = model.load_from_checkpoint(
            cfg.load_state_dict,
            class_labels=datamodule.unique_labels,
            all_labels=datamodule.all_labels,
            example_input_array=example_input.detach().cpu(),
        )

    # log hparam metrics to tensorboard:
    log.info("Logging hparams to tensorboard")
    hydra_params = utils.log_hyperparameters(config=cfg, model=model)
    for this_logger in logger:
        if "tensorboard" in str(this_logger):
            log.info("Add hparams to tensorboard")
            this_logger.log_hyperparams(hydra_params, {"hp/loss": 0, "hp/accuracy": 0, "hp/epoch": 0})
        else:
            this_logger.log_hyperparams(hydra_params)

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to lightning!")
    model.hydra_params = hydra_params

    # Init Trainer:
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks, _convert_="partial")

    # trainer.tune(model, data_module)
    log.info("Starting training")
    trainer.fit(model, datamodule)

    # Print path to best checkpoint
    if trainer.checkpoint_callback.best_model_path is not None:
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
