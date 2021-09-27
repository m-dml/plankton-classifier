import logging

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


class CIFAR10SimClrDataSet(CIFAR10):
    def __init__(self, transform=None, *args, **kwargs):
        super(CIFAR10SimClrDataSet, self).__init__(*args, **kwargs)
        self.transform = transform

        assert self.transform is not None, "Transform must be set"

    def __getitem__(self, index: int):
        img, targets = self.data[index], self.targets[index]

        image = Image.fromarray(img)
        image_copy = image.copy()

        image = self.transform(image)
        image_copy = self.transform(image_copy)

        if torch.equal(image, image_copy):
            self.console_logger.warning(f"Sampled Images are the same at index {index}")

        return (image, image_copy), (torch.tensor(targets), "")


class CIFAR10DataLoader(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        data_base_path: str,
        batch_size: int,
        num_workers: int,
        train_transforms,
        valid_transforms,
        shuffle_train_dataset: bool = True,
        shuffle_valid_dataset: bool = False,
        shuffle_test_dataset: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ):
        super(CIFAR10DataLoader, self).__init__()

        self.cfg_dataset = dataset
        self.data_path = data_base_path

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.shuffle_train_dataset = shuffle_train_dataset
        self.shuffle_valid_dataset = shuffle_valid_dataset
        self.shuffle_test_dataset = shuffle_test_dataset
        self.pin_memory = pin_memory

        # for integration with the plankton structure:
        self.unique_labels = []  # noqa
        self.all_labels = []  # noqa

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_data: Dataset = instantiate(self.cfg_dataset, transform=self.train_transforms, train=True)
            logging.debug(f"Number of training samples: {len(self.train_data)}")
            self.unique_labels = self.train_data.classes

            self.valid_data: Dataset = instantiate(self.cfg_dataset, transform=self.train_transforms, train=False)
            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == "test" or stage is None:
            self.test_data: Dataset = instantiate(self.cfg_dataset, transform=self.train_transforms, train=False)
            self.unique_labels = self.test_data.classes

    def train_dataloader(self):
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
            shuffle=self.shuffle_valid_dataset,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_test_dataset,
            pin_memory=self.pin_memory,
        )
