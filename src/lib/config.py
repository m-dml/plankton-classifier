from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.lib.callbacks import CheckpointCallback, EarlyStoppingCallback, GPUMonitur, LRMonitor
from src.lib.datamodule import (
    CIFAR10DataLoader,
    CIFAR10Dataset,
    CIFAR10DatasetSimClr,
    PlanktonDataLoader,
    PlanktonDataSet,
    PlanktonDataSetSimCLR,
    PlanktonMultiLabelDataLoader,
    PlanktonMultiLabelDataSet,
)
from src.lib.lightning_module import LitModule
from src.lib.logger import MLFlowLogger, TensorBoardLogger, TestTubeLogger
from src.lib.loss import CrossEntropyLoss, KLDivLoss, NLLLoss, NTXentLoss, SimCLRLoss
from src.lib.metrics import Accuracy, MultiLabelAccuracy
from src.lib.model import Classifier, CustomResnet, ResNet
from src.lib.optimizer import LARS, SGD, Adam, RMSprop
from src.lib.pl_plugins import DDPPlugin, SingleDevicePlugin
from src.lib.trainer import Trainer


def register_configs() -> None:
    cs = ConfigStore.instance()
    # my own classes
    cs.store(name="base_lightning_module", node=LitModule, group="lightning_module")

    # the model:
    feature_extractor_group = "model/feature_extractor"
    cs.store(name="resnet_base", node=ResNet, group=feature_extractor_group)
    cs.store(name="custom_resnet_base", node=CustomResnet, group=feature_extractor_group)

    cs.store(name="classifier_base", node=Classifier, group="model/classifier")

    # data:
    cs.store(name="plankton_datamodule_base", node=PlanktonDataLoader, group="datamodule")
    cs.store(name="plankton_datamodule_multilabel_base", node=PlanktonMultiLabelDataLoader, group="datamodule")
    cs.store(name="cifar10_datamodule_base", node=CIFAR10DataLoader, group="datamodule")

    dataset_group = "datamodule/dataset"
    cs.store(name="simclr_base", node=PlanktonDataSetSimCLR, group=dataset_group)
    cs.store(name="default_base", node=PlanktonDataSet, group=dataset_group)
    cs.store(name="multilabel_base", node=PlanktonMultiLabelDataSet, group=dataset_group)
    cs.store(name="cifar10_base", node=CIFAR10Dataset, group=dataset_group)
    cs.store(name="cifar10simclr_base", node=CIFAR10DatasetSimClr, group=dataset_group)

    # external objects:
    cs.store(name="base_trainer", node=Trainer, group="trainer")

    # logger:
    cs.store(name="test_tube", node=TestTubeLogger, group="logger/test_tube")
    cs.store(name="tensorboard", node=TensorBoardLogger, group="logger/tensorboard")
    cs.store(name="ml_flow", node=MLFlowLogger, group="logger/ml_flow")

    # callbacks:
    cs.store(name="model_checkpoint", node=CheckpointCallback, group="callbacks/checkpoint")
    cs.store(name="gpu_monitoring", node=GPUMonitur, group="callbacks/gpu_monitoring")
    cs.store(name="early_stopping", node=EarlyStoppingCallback, group="callbacks/early_stopping")
    cs.store(name="lr_monitor", node=LRMonitor, group="callbacks/lr_monitor")

    # optimizer:
    optimizer_group = "optimizer"
    cs.store(name="adam", node=Adam, group=optimizer_group)
    cs.store(name="sgd", node=SGD, group=optimizer_group)
    cs.store(name="rmsprop", node=RMSprop, group=optimizer_group)
    cs.store(name="lars", node=LARS, group=optimizer_group)

    # loss:
    cs.store(name="nll_loss", node=NLLLoss, group="loss")
    cs.store(name="simclr_loss", node=SimCLRLoss, group="loss")
    cs.store(name="nt_xent_loss", node=NTXentLoss, group="loss")
    cs.store(name="kl_div_loss", node=KLDivLoss, group="loss")
    cs.store(name="cross_entropy_loss", node=CrossEntropyLoss, group="loss")

    # metrics:
    cs.store(name="accuracy", node=Accuracy, group="metric")
    cs.store(name="multi_label_accuracy", node=MultiLabelAccuracy, group="metric")

    # pl training strategies:
    cs.store(name="DDP", node=DDPPlugin, group="strategy")
    cs.store(name="SingleDevice", node=SingleDevicePlugin, group="strategy")

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
    loss: Any = MISSING
    metric: Any = MISSING
    strategy: Any = None

    evaluate: bool = False
    scheduler: Any = None
    random_seed: int = 42
    print_config: bool = True
    debug: bool = False
    ignore_warnings: bool = False
    load_state_dict: Any = None  # if loading from state dict provide path to ckpt file as string here
    output_dir_base_path: str = MISSING
    auto_tune: bool = False  # if true runs trainer.tune() before trainer.fit()
    log_level: str = "info"
