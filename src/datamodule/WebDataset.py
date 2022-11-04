import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import webdataset

from src.utils import utils


class CustomWebDataset(webdataset.WebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def __len__(self):
    #     return -1


class CustomWebLoader(webdataset.WebLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def __len__(self):
    #     return -1


class WebDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        excluded_labels,
        batch_size,
        num_workers,
        train_split,  # The fraction size of the training data
        validation_split,  # The fraction size of the validation data (rest ist test)
        shuffle_train_dataset,  # whether to shuffle the train dataset (bool)
        shuffle_validation_dataset,
        super_classes,  # TODO: implement super classes
        oversample_data,  # TODO: implement oversampling
        random_seed,
        train_transforms,
        valid_transforms,
        data_base_path,
        is_in_simclr_mode,
        label_list,
        subsample_supervised=100,
        shuffle_size=5000,
        *args,
        **kwargs,
    ):
        torch.manual_seed(random_seed)
        super().__init__(*args)

        self.console_logger = utils.get_logger(__name__)
        self.no_augmentations_log_state = False  # Used to log only once if no augmentations are defined
        self.urls = []  # container for all .tar files

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.excluded_labels = excluded_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train_dataset = shuffle_train_dataset
        self.shuffle_validation_dataset = shuffle_validation_dataset
        self.shuffle_size = shuffle_size
        self.data_base_path = data_base_path
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.is_in_simclr_mode = is_in_simclr_mode
        self.train_split = train_split
        self.validation_split = validation_split

        self.subsample_supervised = subsample_supervised  # TODO: implement subsampling
        self.unique_labels = label_list or []
        if not self.is_in_simclr_mode and len(self.unique_labels) <= 1:
            raise ValueError(f"Finetune mode requires at least two labels, but only got {self.unique_labels}. "
                             f"Labels can be set using datamodule.label_list")
        if self.is_in_simclr_mode:
            self.training_class_counts = None

    def prepare_data(self, *args, **kwargs):
        if not os.path.exists(self.data_base_path):
            raise NotADirectoryError(f"Data base path <{self.data_base_path}> does not exist")
        for dir_path, _, filenames in os.walk(self.data_base_path):
            for file in filenames:
                if file.endswith(".tar"):
                    self.urls.append(f"file:{os.path.join(dir_path, file)}")

        num_files = len(self.urls)
        train_files_idx = int(np.ceil(num_files * self.train_split))
        validation_files_idx = int(np.ceil(num_files * self.validation_split)) + train_files_idx

        self.train_data = self.urls[:train_files_idx]
        self.valid_data = self.urls[train_files_idx:validation_files_idx]
        self.test_data = self.urls[validation_files_idx:]

    def make_loader(self, urls, mode="fit"):
        shuffle = 0
        if mode == "fit":
            augmentations: torchvision.transforms = self.train_transforms
            if isinstance(self.shuffle_train_dataset, bool):
                shuffle = self.shuffle_size if self.shuffle_train_dataset else 0

        elif mode == "eval":
            augmentations: torchvision.transforms = self.valid_transforms
            if isinstance(self.shuffle_validation_dataset, bool):
                shuffle = self.shuffle_size if self.shuffle_validation_dataset else 0
        else:
            raise ValueError("Mode must be either 'fit' or 'eval'")

        if self.is_in_simclr_mode:
            dataset = (
                CustomWebDataset(urls)
                .shuffle(shuffle)
                .decode("pil")
                .to_tuple("input.png")
                .map_tuple(lambda x: self.transform(x, augmentations))
                .batched(self.batch_size, partial=False, collation_fn=self.batch_collate_fn_pretrain)
            )

        else:
            dataset = (
                CustomWebDataset(urls)
                .shuffle(shuffle)
                .decode("pil", "txt")
                .tu_tuple("input.png", "label.txt")
                .map_tuple(lambda x: self.transform(x, augmentations), lambda x: x)  # TODO: implement label encoding
                .batched(self.batch_size, pagrtial=False)
            )

        loader = CustomWebLoader(dataset, batch_size=None, shuffle=False, num_workers=self.num_workers)
        return loader

    @staticmethod
    def batch_collate_fn_pretrain(samples):
        assert isinstance(samples[0], (list, tuple)), type(samples[0])
        image_list = []
        image_copy_list = []
        labels = []
        for sample in samples:
            sample = sample[0]
            image_list.append(sample[0][0])
            image_copy_list.append(sample[0][1])
            labels.append(sample[1])
        image = torch.stack(image_list)
        image_copy = torch.stack(image_copy_list)
        labels = torch.stack(labels)
        return (image, image_copy), labels

    def transform(self, image, augmentations):
        image = image
        if self.is_in_simclr_mode:
            image_copy = image.copy()
            if augmentations:
                image = augmentations(image)
                image_copy = augmentations(image_copy)

                if image.equal(image_copy):
                    self.console_logger.warning("Image and image copy are equal")
            else:
                raise ValueError("Transforms must be defined for pretraining")

            return (image, image_copy), (torch.tensor([0]))

        else:
            if augmentations:
                image = augmentations(image)
            elif not self.no_augmentations_log_state:
                self.no_augmentations_log_state = True
                self.console_logger.warning("No augmentations defined. Using original image")
            return image

    def train_dataloader(self):
        return self.make_loader(self.train_data, mode="fit")

    def val_dataloader(self):
        return self.make_loader(self.valid_data, mode="eval")

    def test_dataloader(self):
        return self.make_loader(self.test_data, mode="eval")
