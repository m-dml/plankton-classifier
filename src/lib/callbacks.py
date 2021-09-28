from dataclasses import dataclass


@dataclass
class CheckpointCallback:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    monitor: str = "loss/Validation"
    save_top_k: int = 1
    save_last: bool = True
    mode: str = "min"
    verbose: bool = False
    dirpath: str = "./logs/checkpoints/"  # use  relative path, so it can be adjusted by hydra
    filename: str = "{epoch:02d}"


@dataclass
class GPUMonitur:
    _target_: str = "pytorch_lightning.callbacks.GPUStatsMonitor"


@dataclass
class EarlyStopipingCallback:
    _target_: str = "pytorch_lightning.callbacks.early_stopping.EarlyStopping"
    monitor = "Accuracy/Validation"
    min_delta = 0.00
    patience = 10
    verbose = True
    mode = "max"