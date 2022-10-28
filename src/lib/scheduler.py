from omegaconf import MISSING
from dataclasses import dataclass
from typing import Any


@dataclass
class ReduceLRonPlateau:
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    _recursive_: bool = False
    optimizer: Any = None
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    verbose: bool = True
    threshold: float = 0.0001
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-08


@dataclass
class CosineAnnealingWarmStart:
    _target_: str = "linear_warmup_decay"  # Dummy string


@dataclass
class CosineAnnealingLR:
    _target_: str = "cosine"  # Dummy string