import logging
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models.LightningBaseModel import LightningModel
from src.utils import CONFIG
from src.utils.DataLoader import PlanktonDataLoader
from src.utils.transformations import transform_function

def load_config():
    parser = ArgumentParser()
    parser.add_argument("--config_file", "-f", type=str, default="default_config.yaml",
                        help="Set the configuration file used for the experiment.")

    args = parser.parse_args()
    with open(os.path.abspath(args.config_file), "r") as f:
        config_dict = yaml.safe_load(f)

    # update values in the config class.
    CONFIG.update(config_dict)


def main():
    load_config()
    torch.manual_seed(CONFIG.random_seed)
    pl.seed_everything(CONFIG.random_seed)

    if CONFIG.debug_mode:
        logging.basicConfig(level=logging.DEBUG, format='%(name)s %(funcName)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(name)s %(funcName)s %(levelname)s %(message)s')

    if CONFIG.debug_mode:
        torch.autograd.set_detect_anomaly(True)

    logging.warning(CONFIG.__dict__)  # prints the whole config used for that run

    transform = transform_function(CONFIG.transforms, CONFIG.final_image_size)

    data_module = PlanktonDataLoader.from_argparse_args(CONFIG, transform=transform)
    data_module.setup()

    # if the model is trained on GPU add a GPU logger to see GPU utilization in comet-ml logs:
    if CONFIG.gpus == 0:
        callbacks = None
    else:
        callbacks = [pl.callbacks.GPUStatsMonitor()]

    # logging to tensorboard:
    test_tube_logger = pl_loggers.TestTubeLogger(save_dir=CONFIG.tensorboard_logger_logdir,
                                                 name=CONFIG.experiment_name,
                                                 create_git_tag=False,
                                                 log_graph=True)

    # initializes a callback to save the 5 best model weights measured by the lowest loss:
    checkpoint_callback = ModelCheckpoint(monitor="NLL Validation",
                                          save_top_k=5,
                                          mode='min',
                                          save_last=True,
                                          dirpath=os.path.join(CONFIG.checkpoint_file_path, CONFIG.experiment_name),
                                          )

    model = LightningModel(class_labels=data_module.unique_labels, **CONFIG.__dict__)

    trainer = pl.Trainer.from_argparse_args(CONFIG,
                                            callbacks=callbacks,
                                            logger=[test_tube_logger],
                                            checkpoint_callback=checkpoint_callback,
                                            log_every_n_steps=CONFIG.log_interval,
                                            )

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
