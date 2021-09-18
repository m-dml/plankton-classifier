import glob
import logging
import os
import pathlib
import pickle
import random
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import PIL.PngImagePlugin
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
from torchvision.transforms import transforms
from tqdm import tqdm

from src.utils import utils


class ParentDataSet(Dataset):
    def __init__(self, files, integer_labels, final_image_size=500, transform=None, preload_dataset=False):
        self.files = files
        self.integer_labels = integer_labels

        self.transform = transform
        self.preload_dataset = preload_dataset
        self.final_image_size = final_image_size

        self.console_logger = utils.get_logger(__name__)

    def get_labels(self):  # this is used by the torchsampler
        _, label_names = zip(*self.files)
        labels = [self.integer_labels[label_name] for label_name in label_names]
        return labels

    def load_file(self, file):
        this_image = Image.open(file)
        return this_image

    def __len__(self):
        return len(self.files)

    @abstractmethod
    def __getitem__(self, index) -> (Any, Union[torch.Tensor, List], str):
        return Any, Union[torch.Tensor, List], str


class PlanktonDataSet(ParentDataSet):
    def __init__(self, *args, **kwargs):
        super(PlanktonDataSet, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, label_name = self.files[item]
        if not self.preload_dataset:
            image = self.load_file(image)

        if self.transform:
            image = self.transform(image)

        if isinstance(image, PIL.PngImagePlugin.PngImageFile):
            image = transforms.ToTensor()(image)

        label = torch.Tensor([self.integer_labels[label_name]])

        return image, label, label_name


class PlanktonDataSetSimCLR(ParentDataSet):
    def __init__(self, *args, **kwargs):
        super(PlanktonDataSetSimCLR, self).__init__(*args, **kwargs)
        assert self.transform is not None, "Transform should be set"

    def __getitem__(self, item):
        image, label_name = self.files[item]
        if not self.preload_dataset:
            image = self.load_file(image)
        image_copy = image.copy()

        image = self.transform(image)
        image_copy = self.transform(image_copy)

        if torch.equal(image, image_copy):
            self.console_logger.warning(f"Sampled Images are the same at index {item}")

        return (image, image_copy), torch.tensor(list()), ""


class PlanktonDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        excluded_labels,
        batch_size,
        num_workers,
        train_split,  # The fraction size of the training data
        validation_split,  # The fraction size of the validation data (rest ist test)
        shuffle_train_dataset,  # whether to shuffle the train dataset (bool)
        shuffle_validation_dataset,
        shuffle_test_dataset,
        preload_dataset,
        use_planktonnet_data,
        use_klas_data,
        use_canadian_data,
        super_classes,
        oversample_data,
        klas_data_path,
        planktonnet_data_path,
        canadian_data_path,
        random_seed,
        train_transforms,
        valid_transforms,
        dataset,
        **kwargs,
    ):
        super().__init__()

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.input_channels = None
        self.output_channels = None
        self.unique_labels = []
        self.all_labels = []
        self.integer_class_label_dict = dict()

        self.excluded_labels = excluded_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split  # The fraction size of the training data
        self.validation_split = validation_split  # The fraction size of the validation data (rest ist test)
        self.shuffle_train_dataset = shuffle_train_dataset  # whether to shuffle the train dataset (bool)
        self.shuffle_validation_dataset = shuffle_validation_dataset
        self.shuffle_test_dataset = shuffle_test_dataset
        self.preload_dataset = preload_dataset
        self.klas_data_path = klas_data_path
        self.planktonnet_data_path = planktonnet_data_path
        self.canadian_data_path = canadian_data_path
        self.use_planktonnet_data = use_planktonnet_data
        self.use_klas_data = use_klas_data
        self.use_canadian_data = use_canadian_data
        self.super_classes = super_classes
        self.oversample_data = oversample_data
        self.random_seed = random_seed
        self.cfg_dataset = dataset

    def setup(self, stage=None):
        training_pairs = self.prepare_data_setup()
        self.integer_class_label_dict = self.set_up_integer_class_labels()

        if len(training_pairs) == 0:
            if self.use_klas_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.klas_data_path)}")
            if self.use_canadian_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.canadian_data_path)}")
            if self.use_planktonnet_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.planktonnet_data_path)}")

        if self.use_canadian_data:
            train_subset = training_pairs[0]
            valid_subset = training_pairs[1]
            test_subset = training_pairs[1]  # This is only to not brake the code if a test-dataloader is needed.
        else:
            train_split = self.train_split
            valid_split = train_split + self.validation_split
            length = len(training_pairs)

            train_split_start = 0
            train_split_end = int(length * train_split)
            valid_split_start = train_split_end
            valid_split_end = int(length * valid_split)
            test_split_start = valid_split_end
            test_split_end = length

            train_subset = training_pairs[train_split_start:train_split_end]
            valid_subset = training_pairs[valid_split_start:valid_split_end]
            test_subset = training_pairs[test_split_start:test_split_end]

        if stage == "fit" or stage is None:
            self.train_data: Dataset = instantiate(
                self.cfg_dataset,
                files=train_subset,
                integer_labels=self.integer_class_label_dict,
                transform=self.train_transforms,
            )

            logging.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data: Dataset = instantiate(
                self.cfg_dataset,
                files=valid_subset,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
            )

            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == "test" or stage is None:
            self.test_data: Dataset = instantiate(
                self.cfg_dataset,
                files=test_subset,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
            )

    def prepare_data_setup(self):
        files = []
        if self.use_klas_data:
            for folder in tqdm(glob.glob(os.path.join(self.klas_data_path, "*")), desc="Load Klas data"):
                files += self._add_data_from_folder(folder, file_ext="png")

        if self.use_planktonnet_data:
            for folder in tqdm(glob.glob(os.path.join(self.planktonnet_data_path, "*")), desc="Load planktonNet"):
                files += self._add_data_from_folder(folder, file_ext="jpg")

        if self.use_canadian_data:
            for folder in tqdm(
                glob.glob(os.path.join(self.canadian_data_path, "ringstudy_train", "*")), desc="Load canadian data"
            ):
                files += self._add_data_from_folder(folder, file_ext="png")

            test_files = []
            for folder in tqdm(
                glob.glob(os.path.join(self.canadian_data_path, "ringstudy_test", "*")), desc="Load canadian data"
            ):
                test_files += self._add_data_from_folder(folder, file_ext="png")

        random.seed(self.random_seed)
        random.shuffle(files)
        if self.use_canadian_data:
            return files, test_files

        return files[:100]

    def _add_data_from_folder(self, folder, file_ext="png"):
        files = []
        for file in pathlib.Path(folder).rglob(f"*.{file_ext}"):
            label = os.path.split(folder)[-1]
            label = self._find_super_class(label)
            if label in self.excluded_labels:
                continue
            files.append((self.load_image(file, self.preload_dataset), label))
            self.all_labels.append(label)
            if label not in self.unique_labels:
                self.unique_labels.append(label)

        return files

    def _find_super_class(self, label):
        if self.super_classes is None:
            return label
        else:
            for super_class in self.super_classes.keys():
                if label in self.super_classes[super_class]:
                    label = super_class
                    break
            return label

    def train_dataloader(self):
        if self.oversample_data:
            sampler = ImbalancedDatasetSampler(self.train_data)
            # the imbalanced dataloader only works correctly with 0 workers!
            return DataLoader(
                self.train_data, batch_size=self.batch_size, num_workers=0, pin_memory=True, sampler=sampler
            )
        else:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=self.shuffle_train_dataset,
            )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_validation_dataset,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_test_dataset,
            pin_memory=True,
        )

    @staticmethod
    def load_image(image_file, preload):
        if preload:
            this_image = Image.open(image_file)
            return this_image
        else:
            return image_file

    def set_up_integer_class_labels(self):
        integer_class_labels = dict()
        for i, label in enumerate(self.unique_labels):
            integer_class_labels[label] = i
        return integer_class_labels


class CIFAR10DataSet(Dataset):
    def __init__(self, ds_dict: Dict, indexes: List[int], transform=None, shuffle: bool = True):
        self.ds_dict: Dict = ds_dict
        self.labels: List[str] = ds_dict["labels"]
        self.integer_labels = {lbl: i for i, lbl in enumerate(self.labels)}
        self.indexes: List[Tuple] = [(self.integer_labels[lbl], i) for lbl in self.labels for i in indexes]
        if shuffle:
            random.shuffle(indexes)

        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        label_name, index = self.indexes[item]
        label = self.integer_labels[label_name]
        image = self.ds_dict[label][index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, label_name


class CIFAR10DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        file: str,
        batch_size: int,
        num_workers: int,
        train_split: float,
        valid_split: float,
        train_transforms,
        valid_transforms,
        shuffle_train_dataset: bool = True,
        shuffle_valid_dataset: bool = False,
        shuffle_test_dataset: bool = False,
        random_seed: int = 0,
    ):
        super().__init__()

        self.file = file

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_split = train_split
        self.valid_split = valid_split
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.shuffle_train_dataset = shuffle_train_dataset
        self.shuffle_valid_dataset = shuffle_valid_dataset
        self.shuffle_test_dataset = shuffle_test_dataset
        self.random_seed = random_seed

    def setup(self, stage=None):
        train_indexes, valid_indexes, test_indexes = self.prepare_data_setup()

        if stage == "fit" or stage is None:
            self.train_data = CIFAR10DataSet(
                self.dataset, train_indexes, transform=self.train_transforms, shuffle=self.shuffle_train_dataset
            )
            logging.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data = CIFAR10DataSet(
                self.dataset, valid_indexes, transform=self.valid_transforms, shuffle=self.shuffle_valid_dataset
            )
            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == "test" or stage is None:
            self.test_data = CIFAR10DataSet(
                self.dataset, test_indexes, transform=self.valid_transforms, shuffle=self.shuffle_test_dataset
            )

    def prepare_data_setup(self):
        with open(self.file, "rb") as f:
            self.dataset = pickle.load(f)

        labels = self.dataset["labels"]
        indexes = np.arange(len(self.dataset[random.choice(labels)]))
        length = len(indexes)

        train_split_start = 0
        train_split_end = int(length * self.train_split)
        valid_split_start = train_split_end
        valid_split_end = int(length * self.valid_split)
        test_split_start = valid_split_end
        test_split_end = length

        random.seed(self.random_seed)
        random.shuffle(indexes)
        train_indexes = indexes[train_split_start:train_split_end]
        valid_indexes = indexes[valid_split_start:valid_split_end]
        test_indexes = indexes[test_split_start:test_split_end]
        return train_indexes, valid_indexes, test_indexes
