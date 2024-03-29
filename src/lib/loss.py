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


@dataclass
class NTXentLoss:
    _target_: str = "src.utils.NTXentLoss.NTXentLoss"
    temperature: float = 0.5
    sync_ddp: bool = True


@dataclass
class KLDivLoss:
    _target_: str = "torch.nn.KLDivLoss"
    log_target: bool = False
    reduction: str = "batchmean"
