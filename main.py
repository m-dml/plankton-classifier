import logging
import os
from argparse import ArgumentParser
from datetime import datetime as dt

import hydra
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from src.models.LightningBaseModel import LightningModel
from src.utils.DataLoader import PlanktonDataLoader
from src.utils.SquarePadTransform import SquarePad  # noqa
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config")
def main(cfg: DictConfig):
    logging.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")
    torch.manual_seed(cfg.random_seed)
    pl.seed_everything(cfg.random_seed)

    if cfg.debug_mode:
        logging.basicConfig(level=logging.DEBUG, format='%(name)s %(funcName)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(name)s %(funcName)s %(levelname)s %(message)s')

    if cfg.debug_mode:
        torch.autograd.set_detect_anomaly(True)

    logging.warning(cfg.__dict__)  # prints the whole config used for that run

    transform = transforms.Compose([eval(i) for i in cfg.dataloader.transform])
    data_module = hydra.utils.instantiate(cfg.data_module, transform=transform)
    data_module.setup()

    for batch in data_module.train_dataloader():
        example_input, _, _ = batch
        break

    # if the model is trained on GPU add a GPU logger to see GPU utilization in comet-ml logs:
    if CONFIG.gpus == 0:
        callbacks = None
    else:
        callbacks = [pl.callbacks.GPUStatsMonitor()]

    # logging to tensorboard:
    experiment_name = f"{CONFIG.experiment_name}_{dt.now().strftime('%d%m%YT%H%M%S')}"
    test_tube_logger = pl_loggers.TestTubeLogger(save_dir=CONFIG.tensorboard_logger_logdir,
                                                 name=experiment_name,
                                                 create_git_tag=False,
                                                 log_graph=True)

    # initializes a callback to save the 5 best model weights measured by the lowest loss:
    checkpoint_callback = ModelCheckpoint(monitor="NLL Validation",
                                          save_top_k=5,
                                          mode='min',
                                          save_last=True,
                                          dirpath=os.path.join(CONFIG.checkpoint_file_path, experiment_name),
                                          )

    model = LightningModel(class_labels=data_module.unique_labels,
                           all_labels=data_module.all_labels,
                           example_input_array=example_input,
                           **CONFIG.__dict__)

    trainer = pl.Trainer.from_argparse_args(CONFIG,
                                            callbacks=callbacks,
                                            logger=[test_tube_logger],
                                            checkpoint_callback=checkpoint_callback,
                                            log_every_n_steps=CONFIG.log_interval,
                                            )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
