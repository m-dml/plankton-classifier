from dataclasses import dataclass

from omegaconf import MISSING

from src.lib.optimizer import Optimizer


@dataclass
class LitModule:
    _target_: str = "src.models.LightningBaseModel.LightningModel"
    _recursive_: bool = False
    log_confusion_matrices: bool = True
    log_images: bool = False
    log_tsne_image: bool = True
    optimizer: Optimizer = MISSING
