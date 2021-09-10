from dataclasses import dataclass


@dataclass
class NLLLoss:
    _target_: str = "torch.nn.NLLLoss"


@dataclass
class SimCLRLoss:
    _target_: str = "src.utils.SimCLRLoss.SimCLRLoss"
    temperature: float = 0.5
