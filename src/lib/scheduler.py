from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class CyclicLR:
    _target_: str = "torch.optim.lr_scheduler.CyclicLR"
    base_lr: float = 1e-7
    max_lr: float = 1e-2
    step_size_up: int = 500


@dataclass
class LinearWarmupDecay:
    _target_: str = "torch.optim.lr_scheduler.LambdaLR"
    lr_lambda: Any = MISSING
