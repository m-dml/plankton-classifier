from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class GammaSinusGeneratorDataset:
    _target_: str = "src.datamodules.SinusDataLoader.GammaSinusGeneratorDataset"
    target_mean_offset: float = 1.1
    variance: float = 0.25


@dataclass
class GaussianSinusGeneratorDataset:
    _target_: str = "src.datamodules.SinusDataLoader.GaussianSinusGeneratorDataset"
    std: float = 0.5


@dataclass
class SinusDataLoader:
    _target_: str = "src.datamodules.SinusDataLoader.GenerativeDataLoader"
    _recursive_: bool = False
    batch_size: int = 16
    data_amount: int = int(1e6)
    num_workers: int = 0
    pin_memory: bool = True
    dataset: Any = MISSING


@dataclass
class WikiTrafficDataset:
    _target_: str = "src.datamodules.WikiTrafficDataLoader.WikiTrafficDataset"


@dataclass
class WikiTrafficDataLoader:
    _target_: str = "src.datamodules.WikiTrafficDataLoader.WikiTrafficDataLoader"
    _recursive_: bool = False
    data_file: str = MISSING
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    train_size: float = 0.1
    dataset: Any = MISSING
