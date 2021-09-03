from dataclasses import dataclass


@dataclass
class CheckpointCallback:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    monitor: str = "loss/Validation"
    save_top_k: int = 1
    save_last: bool = True
    mode: str = "min"
    verbose: bool = False
    dirpath: str = "./logs/checkpoints/"
    filename: str = "{epoch:02d}"


@dataclass
class GPUMonitur:
    _target_: str = "pytorch_lightning.callbacks.GPUStatsMonitor"
