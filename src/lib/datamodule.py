from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class WebdataLoader:
    _target_: str = "src.datamodule.WebDataset.WebDataLoader"
    excluded_labels: Any = None
    batch_size: int = MISSING
    num_workers: int = MISSING
    train_split: float = 0.7  # The fraction size of the training data
    validation_split: float = 0.1  # The fraction size of the validation data (rest ist test)
    shuffle_train_dataset: bool = True  # whether to shuffle the train dataset (bool)
    shuffle_validation_dataset: bool = False
    super_classes: Any = None
    oversample_data: bool = True
    random_seed: int = 42
    train_transforms: Any = None
    valid_transforms: Any = None
    data_base_path: str = MISSING
    is_in_simclr_mode: bool = False
    subsample_supervised: float = 1
    shuffle_size: int = 5000
    label_list: Any = None


@dataclass
class ParentDataloader:
    pin_memory: bool = False
    batch_size: int = 16
    num_workers: int = 0


@dataclass
class PlanktonDataSet:
    _target_: str = "src.datamodule.DataLoader.PlanktonDataSet"
    final_image_size: int = 500
    preload_dataset: bool = False
    _convert_: Any = "all"


@dataclass
class PlanktonMultiLabelDataSet:
    _target_: str = "src.datamodule.DataLoader.PlanktonMultiLabelDataSet"
    final_image_size: int = 500
    preload_dataset: bool = False
    _convert_: Any = "all"


@dataclass
class PlanktonInferenceDataSet:
    _target_: str = "src.datamodule.DataLoader.PlanktonInferenceDataSet"
    final_image_size: int = 500
    preload_dataset: bool = False
    _convert_: Any = "all"


@dataclass
class PlanktonDataSetSimCLR:
    _target_: str = "src.datamodule.DataLoader.PlanktonDataSetSimCLR"
    final_image_size: int = 500
    preload_dataset: bool = False
    _convert_: Any = "all"


@dataclass
class PlanktonDataLoader(ParentDataloader):
    _target_: str = "src.datamodule.DataLoader.PlanktonDataLoader"
    _recursive_: bool = False
    train_transforms: Any = MISSING
    valid_transforms: Any = MISSING
    dataset: Any = MISSING
    excluded_labels: Any = None
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
    unlabeled_files_to_append: Any = None
    random_seed: int = 0
    reduce_data: bool = False
    subsample_supervised: float = 1.0  # number of samples per class to use
    find_names_from_folder_structure: bool = False


@dataclass
class PlanktonInferenceDataLoader(PlanktonDataLoader):
    _target_: str = "src.datamodule.DataLoader.PlanktonInferenceDataLoader"
    use_planktonnet_data: bool = False
    use_klas_data: bool = False
    use_canadian_data: bool = False
    oversample_data = False
    data_base_path = "./"
    unlabeled_files_to_append: Any = None


@dataclass
class PlanktonMultiLabelDataLoader(PlanktonDataLoader):
    _target_: str = "src.datamodule.DataLoader.PlanktonMultiLabelDataLoader"
    _recursive_: bool = False
    data_base_path: str = "/gpfs/work/machnitz/plankton_dataset/"
    csv_data_path: str = data_base_path + "human_error2"
    convert_probabilities_to_majority_vote: bool = False


@dataclass
class PlanktonMultiLabelSingleScientistDataLoader(PlanktonDataLoader):
    _target_: str = "src.datamodule.DataLoader.PlanktonMultiLabelSingleScientistDataLoader"
    _recursive_: bool = False
    data_base_path: str = "/gpfs/work/machnitz/plankton_dataset/"
    csv_data_path: str = data_base_path + "human_error2"
    which_expert_label: int = 0


@dataclass
class CIFAR10Dataset:
    _target_: str = "torchvision.datasets.CIFAR10"
    download: bool = True
    root: str = MISSING  # path where to save downloaded data


@dataclass
class CIFAR10DatasetSimClr:
    _target_: str = "src.datamodule.CIFAR10.CIFAR10SimClrDataSet"
    download: bool = True
    root: str = MISSING  # path where to save downloaded data


@dataclass
class CIFAR10DataLoader(ParentDataloader):
    _target_: str = "src.datamodule.CIFAR10.CIFAR10DataLoader"
    _recursive_: bool = False

    dataset: Any = MISSING
    train_transforms: Any = MISSING
    valid_transforms: Any = MISSING
    shuffle_train_dataset: bool = True
    shuffle_validation_dataset: bool = False
    shuffle_test_dataset: bool = False
    data_base_path: str = "/gpfs/work/zinchenk/cifar10-dataset"
