import glob
import logging
import os

from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision

from src.utils import CONFIG


class PlanktonDataSet(Dataset):
    def __init__(self, files, transform=None):
        self.files = files

        self.transform = transform

        if len(self.files) > 0:
            self.example_x = self.load_and_transform_file(self.files[0][0])
        else:
            self.example_x = None

    def __getitem__(self, item):
        # todo: write data fetching (make it possible to preload and to load lazy!)
        return

    def load_and_transform_file(self, file):
        # todo: write data loading

        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    def get_input_channel_size(self) -> int:
        # todo: return right channel size
        raise NotImplementedError

    def get_output_channel_size(self) -> int:
        # todo: return right channel size
        raise NotImplementedError

    def __len__(self):
        # todo: return right length for 1. lazy loading 2. preloaded data
        raise NotImplementedError


class PlanktonDataLoader(pl.LightningDataModule):
    def __init__(self, transform=None):
        super().__init__()

        self.transform = transform

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.input_channels = None
        self.output_channels = None

        self.batch_size = CONFIG.batch_size
        self.num_workers = CONFIG.num_workers
        self.train_split = CONFIG.train_split  # The fraction size of the training data
        self.validation_split = CONFIG.validation_split  # The fraction size of the validation data (rest ist test)
        self.shuffle_train_dataset = CONFIG.shuffle_train_dataset  # whether to shuffle the train dataset (bool)
        self.shuffle_validation_dataset = CONFIG.shuffle_validation_dataset
        self.shuffle_test_dataset = CONFIG.shuffle_test_dataset
        self.preload_dataset = CONFIG.preload_dataset
        self.old_data_path = os.path.join(CONFIG.plankton_data_base_path, CONFIG.old_sorted_plankton_data)
        self.new_data_path = os.path.join(CONFIG.plankton_data_base_path, CONFIG.new_sorted_plankton_data)

        self.use_old_data = CONFIG.use_old_data
        self.use_new_data = CONFIG.use_new_data
        self.use_only_subclasses_of_old_data = CONFIG.use_only_subclasses_of_old_data
        self.preload_dataset = CONFIG.preload_dataset

    def setup(self, stage=None):
        training_pairs = self.prepare_data_setup()

        if len(training_pairs) == 0:
            raise FileNotFoundError(f"Did not find any files")

        train_split = self.train_split
        valid_split = train_split + self.validation_split
        length = len(training_pairs)

        train_split_start = 0
        train_split_end = int(length * train_split)
        valid_split_start = train_split_end
        valid_split_end = int(length * valid_split)
        test_split_start = valid_split_end
        test_split_end = length

        train_subset = training_pairs[train_split_start: train_split_end]
        valid_subset = training_pairs[valid_split_start: valid_split_end]
        test_subset = training_pairs[test_split_start: test_split_end]

        if stage == 'fit' or stage is None:
            self.train_data = PlanktonDataSet(train_subset, transform=self.transform)
            self.input_channels = self.train_data.get_input_channel_size()
            self.output_channels = self.train_data.get_output_channel_size()
            logging.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data = PlanktonDataSet(valid_subset, transform=self.transform)
            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == 'test' or stage is None:
            self.test_data = PlanktonDataSet(test_subset, transform=self.transform)
            self.input_channels = self.train_data.get_input_channel_size()
            self.output_channels = self.train_data.get_output_channel_size()

    def prepare_data_setup(self):
        files = []
        if self.use_old_data:
            for folder in glob.glob(os.path.join(self.old_data_path, "*")):
                if not self.use_only_subclasses_of_old_data:
                    raw_file_paths = glob.glob(folder + "*/*/*.tif")
                    for file in raw_file_paths:
                        files.append((load_image(file, self.preload_dataset), os.path.split(folder)[-1]))

                else:
                    for sub_folder in glob.glob(os.path.join(folder, "*")):
                        raw_file_paths = glob.glob(sub_folder + "*/*.tif")
                        for file in raw_file_paths:
                            files.append((load_image(file, self.preload_dataset), os.path.split(folder)[-1]))

        if self.use_new_data:
            for folder in glob.glob(os.path.join(self.new_data_path, "*")):
                raw_file_paths = glob.glob(folder + "*/*/*.tif")
                for file in raw_file_paths:
                    files.append((load_image(file, self.preload_dataset), os.path.split(folder)[-1]))
        return files

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_train_dataset, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_validation_dataset, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_test_dataset, pin_memory=True)


def load_image(image_file, preload):
    if preload:
        this_image = Image.open(image_file)
        return torchvision.transforms.ToTensor()(this_image)
    else:
        return image_file