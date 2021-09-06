from dataclasses import dataclass


@dataclass
class Optimizer:
    lr: float = 0.02


@dataclass
class Adam(Optimizer):
    _target_: str = "torch.optim.Adam"


@dataclass
class SGD(Optimizer):
    _target_: str = "torch.optim.SGD"
    momentum: float = 0


@dataclass
class RMSprop(Optimizer):
    _target_: str = "torch.optim.RMSprop"
    momentum: float = 0
