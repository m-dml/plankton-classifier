from dataclasses import dataclass


@dataclass
class NLLLoss:
    _target_: str = "torch.nn.NLLLoss"


@dataclass
class CrossEntropyLoss:
    _target_: str = "torch.nn.CrossEntropyLoss"


@dataclass
class SimCLRLoss:
    _target_: str = "src.utils.SimCLRLoss.SimCLRLoss"
    temperature: float = 0.5
