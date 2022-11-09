from dataclasses import dataclass

from omegaconf import MISSING

from src.lib.optimizer import Optimizer


@dataclass
class LitModule:
    _target_: str = "src.models.lightning_module.LightningModel"
    _recursive_: bool = False
    log_confusion_matrices: bool = False
    log_images: bool = False
    log_tsne_image: bool = False
    optimizer: Optimizer = MISSING
    freeze_feature_extractor: bool = False
    temperature_scale: bool = False
