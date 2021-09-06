from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ResnetBaseClass:
    num_classes: int = 1000  # has to be 1000 for pretrained model
    pretrained: bool = True


@dataclass
class resnet18(ResnetBaseClass):
    _target_: str = "torchvision.models.resnet18"


@dataclass
class ResNet:
    _target_: str = "src.models.ResNet.ResNet"
    resnet: ResnetBaseClass = MISSING
