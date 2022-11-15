import hydra
import torch
from omegaconf import DictConfig, open_dict
import pytorch_lightning as pl
from src.utils import utils


def init_callbacks(cfg) -> list[pl.callbacks.base.Callback]:
    # Init Lightning callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def init_logger(cfg) -> list[pl.loggers.base.LightningLoggerBase]:
    # Init Lightning loggers
    cfg_logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                cfg_logger.append(hydra.utils.instantiate(lg_conf))
    return cfg_logger


def init_datamodule(cfg, training_class_counts=None) -> pl.LightningDataModule:
    train_transforms = hydra.utils.instantiate(cfg.datamodule.train_transforms)
    valid_transforms = hydra.utils.instantiate(cfg.datamodule.valid_transforms)

    if "dataset" not in cfg.datamodule.keys():
        with open_dict(cfg.datamodule):
            cfg.datamodule.dataset = None

    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        train_transforms=train_transforms,
        valid_transforms=valid_transforms,
        dataset=cfg.datamodule.dataset,
        is_ddp=False,
    )

    datamodule.training_class_counts = training_class_counts

    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    return datamodule


def get_number_of_stepping_batches(cfg, datamodule) -> int:
    # get number of training samples_per_device and epoch:
    datamodule.is_ddp = False
    try:
        stepping_batches = len(datamodule.train_dataloader())
    except TypeError:
        stepping_batches = cfg.trainer.max_steps

    return stepping_batches


def generate_example_input_array(datamodule) -> torch.Tensor:
    # generate example input array:
    example_input = None
    for batch in datamodule.val_dataloader():
        example_input, _ = batch
        if isinstance(example_input, (tuple, list)):
            example_input = torch.stack(example_input).detach().cpu()
        break

    assert example_input is not None, "example_input is still None."
    return example_input


def init_model(cfg, datamodule, stepping_batches, example_input, pretrain) -> pl.LightningModule:
    model = hydra.utils.instantiate(
        cfg.lightning_module,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        feature_extractor=cfg.model.feature_extractor,
        classifier=cfg.model.classifier,
        loss=cfg.loss,
        metric=cfg.metric,
        is_in_simclr_mode=pretrain,
        batch_size=cfg.datamodule.batch_size,
        num_unique_labels=len(datamodule.unique_labels),
        num_steps_per_epoch=stepping_batches,
    )

    model.set_external_data(
        class_labels=datamodule.unique_labels,
        example_input_array=example_input.detach().cpu(),
    )
    return model


def log_hparams(cfg, cfg_logger, model) -> pl.LightningModule:
    # log hparam metrics to tensorboard:
    hydra_params = utils.log_hyperparameters(config=cfg, model=model)
    for this_logger in cfg_logger:
        if "tensorboard" in str(this_logger):
            this_logger.log_hyperparams(hydra_params, {"hp/loss": 0, "hp/accuracy": 0, "hp/epoch": 0})
        else:
            this_logger.log_hyperparams(hydra_params)

    # Send some parameters from config to all lightning loggers
    model.hydra_params = hydra_params
    return model


def init_trainer(cfg, cfg_logger, callbacks, stepping_batches) -> pl.Trainer:
    # Init Trainer:
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        strategy=cfg.strategy,
        logger=cfg_logger,
        callbacks=callbacks,
        _convert_="partial",
        profiler=cfg.profiler,
    )

    if cfg.trainer.val_check_interval:
        if cfg.trainer.val_check_interval > stepping_batches:
            trainer.val_check_interval = 1.0

    return trainer


def create_and_run(cfg: DictConfig, training_class_counts=None) -> float:
    callbacks = init_callbacks(cfg)
    cfg_logger = init_logger(cfg)
    datamodule = init_datamodule(cfg, training_class_counts=training_class_counts)

    stepping_batches = get_number_of_stepping_batches(cfg, datamodule)
    example_input = generate_example_input_array(datamodule)

    model = init_model(cfg, datamodule, stepping_batches, example_input, pretrain=cfg.pretrain)
    model = log_hparams(cfg, cfg_logger, model)

    trainer = init_trainer(cfg, cfg_logger, callbacks, stepping_batches)
    trainer.fit(model, datamodule)

    return trainer.callback_metrics["hp/loss"].item()
