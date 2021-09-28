from dataclasses import dataclass


@dataclass
class CyclicLR:
    _target_: str = "torch.optim.lr_scheduler.CyclicLR"
    base_lr: float =  1e-7
    max_lr: float = 1e-2