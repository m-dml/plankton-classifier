import glob
import logging
import os
import pathlib
import random
from abc import abstractmethod
from typing import Any, List, Union

import numpy as np
import pandas as pd
import PIL.PngImagePlugin
import pytorch_lightning as pl
import torch
from catalyst.data.sampler import BalanceClassSampler, DistributedSamplerWrapper
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
        if isinstance(label_name, list) and len(label_name) > 1:
            label = torch.Tensor(label_name)
        else:
            label = torch.Tensor([self.integer_labels[label_name]])

        return image, (label, label_name)


class PlanktonMultiLabelDataSet(ParentDataSet):
    def __init__(self, *args, **kwargs):
        super(PlanktonMultiLabelDataSet, self).__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, label_probabilities, labels = self.files[item]
        if not self.preload_dataset:
            image = self.load_file(image)

        if self.transform:
            image = self.transform(image)

        if isinstance(image, PIL.PngImagePlugin.PngImageFile):
            image = transforms.ToTensor()(image)

        return image, (torch.tensor(label_probabilities), torch.tensor(labels))


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

        return (image, image_copy), (torch.tensor(self.integer_labels[label_name]))


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
        reduce_data,
        pin_memory=False,
        unlabeled_files_to_append=None,
        is_ddp=False,
        subsample_supervised=100,
        **kwargs,
    ):
        super().__init__()

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.train_labels = None
        self.valid_labels = None
        self.test_labels = None

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
        self.unlabeled_files_to_append = unlabeled_files_to_append

        self.use_planktonnet_data = use_planktonnet_data
        self.use_klas_data = use_klas_data
        self.use_canadian_data = use_canadian_data

        self.super_classes = super_classes
        self.oversample_data = oversample_data
        self.random_seed = random_seed
        self.cfg_dataset = dataset
        self.pin_memory = pin_memory
        self.reduce_data = reduce_data
        self.console_logger = utils.get_logger(__name__)
        self.is_ddp = is_ddp
        self.subsample_supervised = subsample_supervised
        self.training_class_counts = None  # how often does each class exist in the training dataset
        self.max_label_value = 0
        self.len_train_data = None

        if self.use_canadian_data:
            raise ValueError("Usage of the Canadian data is not permitted for the paper.")
        if self.use_planktonnet_data:
            raise ValueError("Usage of the Planktonnet data is not permitted for the paper")

        self.console_logger.info("Successfully set up datamodule.")

    def setup(self, stage=None):
        self.console_logger.debug("Loading Training data")
        train_subset = self.prepare_data_setup(subset="train")
        self.console_logger.debug(f"len(train_subset) = {len(train_subset)}")
        self.console_logger.debug("Loading Validation data")
        valid_subset = self.prepare_data_setup(subset="val")
        self.console_logger.debug(f"len(valid_subset) = {len(valid_subset)}")
        self.console_logger.debug("Loading Test data")
        test_subset = self.prepare_data_setup(subset="test")
        self.console_logger.debug(f"len(test_subset) = {len(test_subset)}")

        if self.unlabeled_files_to_append:
            self.console_logger.debug("Trying to load unlabeled files")
            if isinstance(train_subset, str):
                train_subset += self.add_all_images_from_all_subdirectories(self.unlabeled_files_to_append)
            else:
                for folder_with_unlabeled_files in self.unlabeled_files_to_append:
                    train_subset += self.add_all_images_from_all_subdirectories(folder_with_unlabeled_files)

        self.integer_class_label_dict = self.set_up_integer_class_labels()

        if len(train_subset) == 0:
            if self.use_klas_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.klas_data_path)}")
            if self.use_canadian_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.canadian_data_path)}")
            if self.use_planktonnet_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.planktonnet_data_path)}")
        self.len_train_data = int(len(train_subset) / self.batch_size)

        self.console_logger.debug("Separating labels from images")
        self.unique_labels, self.train_labels = np.unique(list(list(zip(*train_subset))[1]), return_inverse=True)
        self.console_logger.debug(f"There are {len(self.unique_labels)} unique training labels: {self.unique_labels}")
        unique_val_labels, self.valid_labels = np.unique(list(list(zip(*valid_subset))[1]), return_inverse=True)
        self.console_logger.debug(f"There are {len(unique_val_labels)} unique validation labels: {unique_val_labels}")
        unique_test_labels, self.test_labels = np.unique(list(list(zip(*test_subset))[1]), return_inverse=True)
        self.console_logger.debug(f"There are {len(unique_test_labels)} unique test labels: {unique_test_labels}")

        # self._test_label_consistency(self.unique_labels, unique_val_labels)
        # self._test_label_consistency(self.unique_labels, unique_test_labels)

        # if self.subsample_supervised:
        #     label_dict = {
        #         label: np.arange(len(self.train_labels))[self.train_labels == label].tolist()
        #         for label in self.unique_labels.flatten()
        #     }
        #     indices = []
        #     for key in sorted(label_dict):
        #         if len(label_dict[key]) >= self.subsample_supervised:
        #             indices += np.random.choice(label_dict[key], self.subsample_supervised, replace=False).tolist()
        #         else:
        #             indices += label_dict[key]
        #     train_subset = [train_subset[i] for i in indices]
        #     self.train_labels = self.train_labels[indices]

        self.console_logger.debug("Getting the image counts for each label")
        _, self.training_class_counts = np.unique(self.train_labels, return_counts=True)

        self.console_logger.info(f"There are {len(train_subset)} training files")
        self.console_logger.info(f"There are {len(valid_subset)} validation files")
        if stage == "fit":
            self.console_logger.info(f"Instantiating training dataset <{self.cfg_dataset._target_}>")
            self.train_data: Dataset = instantiate(
                self.cfg_dataset,
                files=train_subset,
                integer_labels=self.integer_class_label_dict,
                transform=self.train_transforms,
                preload_dataset=self.preload_dataset,
            )

            self.console_logger.info(f"Instantiating validation dataset <{self.cfg_dataset._target_}>")
            self.console_logger.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data: Dataset = instantiate(
                self.cfg_dataset,
                files=valid_subset,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )

            self.console_logger.debug(f"Number of validation samples: {len(self.valid_data)}")

        elif stage == "test":
            self.console_logger.info(f"Instantiating test dataset <{self.cfg_dataset._target_}>")
            self.test_data: Dataset = instantiate(
                self.cfg_dataset,
                files=test_subset,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )

        else:
            raise ValueError(f'<stage> needs to be either "fit" or "test", but is {stage}')

    def prepare_data_setup(self, subset):
        files = []
        if self.use_klas_data:
            for folder in tqdm(glob.glob(os.path.join(self.klas_data_path, subset, "*")), desc="Load Klas data"):
                files += self._add_data_from_folder(folder, file_ext="png")
        return files

    def add_all_images_from_all_subdirectories(self, folder, file_ext="png", recursion_depth=0):
        self.console_logger.debug(f"folder: {folder}")
        self.console_logger.debug(f"Recursion depth: {recursion_depth}")

        all_sys_elements = glob.glob(os.path.join(folder, "*"))

        if self._is_image_folder(folder):
            folder_files = self._add_data_from_folder(folder, file_ext=file_ext)
            self.console_logger.debug("=====================================")
            return folder_files

        files = []
        # if this folder does not contain images, check if it contains other folders:
        for sys_element in all_sys_elements:
            if os.path.isdir(sys_element):
                # using recursion to reach all subdirectories:
                files += self.add_all_images_from_all_subdirectories(
                    sys_element, file_ext, recursion_depth=recursion_depth + 1
                )
        self.console_logger.debug(f"len files {len(files)}")
        self.console_logger.debug("=====================================")
        return files

    @staticmethod
    def _is_image_folder(folder, file_ext="png") -> bool:
        all_files = glob.glob(os.path.join(folder, "*"))
        img_files = glob.glob(os.path.join(folder, "*" + file_ext))

        if (len(all_files) / 2) < len(img_files):  # if more than half of the files in the folder are images:
            return True
        else:
            return False

    def _add_data_from_folder(self, folder, file_ext="png"):
        files = []
        for file in tqdm(pathlib.Path(folder).rglob(f"*.{file_ext}"), position=0, leave=True):
            label = os.path.split(folder)[-1]
            label = self._find_super_class(label)
            if label in self.excluded_labels:
                continue
            files.append((self.load_image(file, self.preload_dataset), label))
            self.all_labels.append(label)
            if label not in self.unique_labels:
                self.unique_labels.append(label)

        if len(files) > 10000 and self.reduce_data:
            self.console_logger.info(f"using only 10k {folder} images from orig={len(files)}")
            return files[:10000]
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
            sampler = BalanceClassSampler(self.train_labels, mode=self.subsample_supervised)
            if self.is_ddp:
                self.console_logger.info("Wrapping the sampler to be ddp compatible")
                sampler = self.ddp_wrap_sampler(sampler)
                self.console_logger.info("Wrapping was successful")

            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=sampler,
            )
        else:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle_train_dataset,
            )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_validation_dataset,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_test_dataset,
            pin_memory=self.pin_memory,
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

    @staticmethod
    def ddp_wrap_sampler(sampler):
        return DistributedSamplerWrapper(sampler)

    @staticmethod
    def _test_label_consistency(labels_a, labels_b):
        if len(labels_a.flatten()) == len(labels_b.flatten()):
            return
        else:
            raise ValueError("Training labels are not the same as validation or test labels!")


class PlanktonMultiLabelDataLoader(PlanktonDataLoader):
    def __init__(
        self,
        human_error2_data_path,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.human_error2_data_path = human_error2_data_path

    def setup(self, stage=None):
        train_subset = self.prepare_data_setup("train")
        valid_subset = self.prepare_data_setup("val")
        test_subset = self.prepare_data_setup("test")

        self.len_train_data = int(len(train_subset) / self.batch_size)


        self.train_labels = list(zip(*train_subset))[1]
        self.valid_labels = list(zip(*valid_subset))[1]
        self.test_labels = list(zip(*test_subset))[1]

        self.console_logger.info(f"There are {len(train_subset)} training files")
        self.console_logger.info(f"There are {len(valid_subset)} validation files")
        if stage == "fit":
            self.console_logger.info(f"Instantiating training dataset <{self.cfg_dataset._target_}>")
            self.train_data: Dataset = instantiate(
                self.cfg_dataset,
                files=train_subset,
                integer_labels=self.train_labels,
                transform=self.train_transforms,
                preload_dataset=self.preload_dataset,
            )

            self.console_logger.info(f"Instantiating validation dataset <{self.cfg_dataset._target_}>")
            self.console_logger.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data: Dataset = instantiate(
                self.cfg_dataset,
                files=valid_subset,
                integer_labels=self.valid_labels,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )

            self.console_logger.debug(f"Number of validation samples: {len(self.valid_data)}")

        elif stage == "test":
            self.console_logger.info(f"Instantiating test dataset <{self.cfg_dataset._target_}>")
            self.test_data: Dataset = instantiate(
                self.cfg_dataset,
                files=test_subset,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )

        else:
            raise ValueError(f'<stage> needs to be either "fit" or "test", but is {stage}')

    def load_multilable_dataset(self, data_path, csv_file):
        df = pd.read_csv(csv_file)
        df = df.drop(columns="Unnamed: 0")
        repl_column_names = dict()
        for column in df.columns:
            column_new = column.strip().lower()
            column_new = column_new[3:] if column_new.startswith("00_") else column_new
            repl_column_names[column] = column_new
            if not column == "file":
                df[column] = df[column].astype("category").cat.codes.astype(int)

        df = df.rename(columns=repl_column_names)

        files = []
        self.max_label_value = df.drop(labels="file", axis=1).max().max()
        self.unique_labels = np.arange(0, self.max_label_value + 1)
        for file, labels in df.set_index("file").iterrows():
            files.append(
                (
                    self.load_image(os.path.join(data_path, file), preload=self.preload_dataset),
                    self.multi_labels_to_probabilities(labels.values, max_label_value=self.max_label_value),
                    labels.values,
                )
            )

        return files

    def prepare_data_setup(self, subset):
        csv_file = os.path.join(self.human_error2_data_path, f"multi_label_{subset}.csv")
        folder = os.path.join(self.human_error2_data_path, subset)
        files = self.load_multilable_dataset(folder, csv_file)

        return files

    @staticmethod
    def multi_labels_to_probabilities(labels, max_label_value):
        n_bins = len(np.arange(0, max_label_value))
        probabilities = np.histogram(labels, bins=n_bins + 1, range=(0, n_bins))[0] / len(labels)
        return probabilities
