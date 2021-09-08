from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class ResNet:
    _target_: str = "torchvision.models.resnet18"
    num_classes: int = 1000  # has to be 1000 for pretrained model
    pretrained: bool = False


@dataclass
class SimCLRFeatureExtractor:
    _target_: str = "src.models.BaseModels.SimCLRFeatureExtractor"
    model: Any = MISSING


@dataclass
class Classifier:
    _target_: str = "src.models.BaseModels.Classifier"
    hidden_layers: list = (1000, 1000)
    activation: Any = MISSING
    input_features: int = 1000
