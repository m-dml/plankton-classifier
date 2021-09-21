from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class PlanktonDataSet:
    _target_: str = "src.datamodule.DataLoader.PlanktonDataSet"
    final_image_size: int = 500
    preload_dataset: bool = False


@dataclass
class PlanktonDataSetSimCLR:
    _target_: str = "src.datamodule.DataLoader.PlanktonDataSetSimCLR"
    final_image_size: int = 500
    preload_dataset: bool = False


@dataclass
class PlanktonDataLoader:
    _target_: str = "src.datamodule.DataLoader.PlanktonDataLoader"
    _recursive_: bool = False
    train_transforms: Any = MISSING
    valid_transforms: Any = MISSING
    dataset: Any = MISSING

    excluded_labels: Any = None
    batch_size: int = 16
    num_workers: int = 0
    train_split: float = 0.8
    validation_split: float = 0.1
    shuffle_train_dataset: bool = True
    shuffle_validation_dataset: bool = False
    shuffle_test_dataset: bool = False
    preload_dataset: bool = False
    use_planktonnet_data: bool = False
    use_klas_data: bool = True
    use_canadian_data: bool = False
    super_classes: Any = None
    oversample_data: bool = True
    data_base_path: str = "/gpfs/work/machnitz/plankton_dataset/"
    klas_data_path: str = data_base_path + "new_data/4David/M160/Sorted"
    planktonnet_data_path: str = data_base_path + "PlanktonNet/DYB-PlanktonNet_V1.0_EN"
    canadian_data_path: str = data_base_path + "canadian_dataset"
    random_seed: int = 0


@dataclass
class CIFAR10Dataset:
    _target_: str = "torchvision.CIFAR10.CIFAR10"
    download: bool = True
    root: str = MISSING  # path where to save downloaded data


@dataclass
class CIFAR10DatasetSimClr:
    _target_: str = "src.datamodule.CIFAR10.CIFAR10SimClrDataSet"
    download: bool = True
    root: str = MISSING  # path where to save downloaded data


@dataclass
class CIFAR10DataLoader:
    _target_: str = "src.datamodule.CIFAR10.CIFAR10DataLoader"
    _recursive_: bool = False

    dataset: Any = MISSING
    train_transforms: Any = MISSING
    valid_transforms: Any = MISSING
    batch_size: int = 16
    num_workers: int = 0
    shuffle_train_dataset: bool = True
    shuffle_validation_dataset: bool = False
    shuffle_test_dataset: bool = False
    data_base_path: str = "/gpfs/work/zinchenk/cifar10-dataset"
