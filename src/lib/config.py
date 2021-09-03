from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.lib.callbacks import CheckpointCallback, GPUMonitur
from src.lib.datamodule import PlanktonDataLoader
from src.lib.lightning_module import LitModule
from src.lib.logger import MLFlowLogger, TensorBoardLogger, TestTubeLogger
from src.lib.model import ResNet, resnet18
from src.lib.optimizer import SGD, Adam, RMSprop
from src.lib.trainer import Trainer


def register_configs() -> None:
    cs = ConfigStore.instance()
    # my own classes
    cs.store(name="base_lightning_module", node=LitModule, group="lightning_module")

    # the model:
    cs.store(name="resnet_base", node=ResNet, group="model")
    cs.store(name="resnet18", node=resnet18, group="model/resnet")

    # data:
    cs.store(name="plankton_datamodule_base", node=PlanktonDataLoader, group="datamodule")

    # external objects:
    cs.store(name="base_trainer", node=Trainer, group="trainer")

    # logger:
    cs.store(name="test_tube", node=TestTubeLogger, group="logger/test_tube")
    cs.store(name="tensorboard", node=TensorBoardLogger, group="logger/tensorboard")
    cs.store(name="ml_flow", node=MLFlowLogger, group="logger/ml_flow")

    # callbacks:
    cs.store(name="model_checkpoint", node=CheckpointCallback, group="callbacks/checkpoint")
    cs.store(name="gpu_monitoring", node=GPUMonitur, group="callbacks/gpu_monitoring")

    # optimizer:
    cs.store(name="adam", node=Adam, group="optimizer")
    cs.store(name="sgd", node=SGD, group="optimizer")
    cs.store(name="rmsprop", node=RMSprop, group="optimizer")

    # register the base config class (this name has to be called in config.yaml):
    cs.store(name="base_config", node=Config)


@dataclass
class Config:
    lightning_module: LitModule = MISSING
    model: Any = MISSING
    datamodule: Any = MISSING
    trainer: Trainer = MISSING
    logger: Any = MISSING
    callbacks: Any = MISSING
    optimizer: Any = MISSING

    random_seed: int = 42
    print_config: bool = True
    debug: bool = False
    ignore_warnings: bool = False
