import glob
import logging
import os
import random

import pytorch_lightning as pl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from src.utils import CONFIG
import PIL
from PIL import Image


class PlanktonDataSet(Dataset):
    def __init__(self, files, integer_labels, final_image_size=500, transform=None):
        self.files = files
        self.integer_labels = integer_labels

        self.transform = transform
        self.preload_dataset = CONFIG.preload_dataset
        self.final_image_size = final_image_size
        self.use_image_morphings = CONFIG.use_image_morphings

    def __getitem__(self, item):
        image, label_name = self.files[item]
        if not self.preload_dataset:
            image = self.load_file(image)

        if self.use_image_morphings:
            sitk_image = self._get_sitk_image(image)
            grayscale_eroded = self._get_image_from_sitk(self._grayscale_erode(sitk_image))
            canny_edges = self._get_image_from_sitk(self._canny_edges(sitk_image))
            grayscale_dilated = self._get_image_from_sitk(self._grayscale_dilated(sitk_image))

        if self.transform:
            image = self.transform(image)
            image = image / 255

            if self.use_image_morphings:
                grayscale_eroded = self.transform(grayscale_eroded)
                canny_edges = self.transform(canny_edges)
                grayscale_dilated = self.transform(grayscale_dilated)

        image = torch.cat([image, grayscale_eroded, grayscale_dilated, canny_edges], dim=0)
        label = torch.Tensor([self.integer_labels[label_name]])

        return image, label, label_name

    def load_file(self, file):
        this_image = Image.open(file)
        return this_image

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _get_sitk_image(image: PIL.Image):
        image_array = np.array(image)
        image_sitk = sitk.GetImageFromArray(image_array)
        return image_sitk

    @staticmethod
    def _get_image_from_sitk(sitk_image) -> PIL.Image:
        array = sitk.GetArrayFromImage(sitk_image)
        pil_image = Image.fromarray(array.astype('uint8'), 'RGB')
        return pil_image

    @staticmethod
    def _grayscale_erode(image_sitk):
        grayscale_erode_filter = sitk.GrayscaleErodeImageFilter()
        grayscale_erode_filter.SetKernelRadius(3)
        grayscale_erode_filter.SetKernelType(sitk.sitkBall)
        grayscale_eroded_image = grayscale_erode_filter.Execute(image_sitk)
        return grayscale_eroded_image

    @staticmethod
    def _grayscale_dilated(image_sitk):
        grayscale_dilate_filter = sitk.GrayscaleDilateImageFilter()

        grayscale_dilate_filter.SetKernelRadius(3)
        grayscale_dilate_filter.SetKernelType(sitk.sitkBall)
        grayscale_dilate_image = grayscale_dilate_filter.Execute(image_sitk)
        return grayscale_dilate_image

    @staticmethod
    def _canny_edges(image_sitk):
        canny_filter = sitk.CannyEdgeDetectionImageFilter()
        float_image = sitk.Cast(image_sitk, sitk.sitkFloat32)
        canny_filter.SetLowerThreshold(20)
        canny_filter.SetVariance(15)
        canny_edges = canny_filter.Execute(float_image)
        return canny_edges


class PlanktonDataLoader(pl.LightningDataModule):
    def __init__(self, transform=None):
        super().__init__()

        self.transform = transform

        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.input_channels = None
        self.output_channels = None
        self.unique_labels = []
        self.all_labels = []
        self.integer_class_labels = dict()

        self.excluded_labels = CONFIG.excluded_classes
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
        self.use_subclasses = CONFIG.use_subclasses
        self.preload_dataset = CONFIG.preload_dataset
        self.super_classes = CONFIG.super_classes

        self.final_image_size = CONFIG.final_image_size

    def setup(self, stage=None):
        training_pairs = self.prepare_data_setup()
        self.integer_class_labels = self.set_up_integer_class_labels()

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
            self.train_data = PlanktonDataSet(train_subset, transform=self.transform,
                                              integer_labels=self.integer_class_labels)
            logging.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data = PlanktonDataSet(valid_subset, transform=self.transform,
                                              integer_labels=self.integer_class_labels)
            logging.debug(f"Number of validation samples: {len(self.valid_data)}")

        if stage == 'test' or stage is None:
            self.test_data = PlanktonDataSet(test_subset, transform=self.transform,
                                             integer_labels=self.integer_class_labels)

    def prepare_data_setup(self):
        excluded = self.excluded_labels
        files = []
        if self.use_old_data:
            for folder in tqdm(glob.glob(os.path.join(self.old_data_path, "*")), desc="Load old data"):
                if not self.use_subclasses:
                    raw_file_paths = glob.glob(folder + "*/*/*.tif")
                    for file in raw_file_paths:
                        label = os.path.split(folder)[-1]
                        if label in excluded:
                            continue
                        files.append((self.load_image(file, self.preload_dataset),label))
                        self.all_labels.append(label)
                        if label not in self.unique_labels:
                            self.unique_labels.append(label)

                else:
                    for sub_folder in tqdm(glob.glob(os.path.join(folder, "*")), desc="Load old data"):
                        raw_file_paths = glob.glob(sub_folder + "*/*.tif")
                        for file in raw_file_paths:
                            label = os.path.split(folder)[-1]
                            if label in excluded:
                                continue
                            files.append((self.load_image(file, self.preload_dataset), label))
                            self.all_labels.append(label)
                            if label not in self.unique_labels:
                                self.unique_labels.append(label)

        if self.use_new_data:
            for folder in tqdm(glob.glob(os.path.join(self.new_data_path, "*")), desc="Load new data"):
                raw_file_paths = glob.glob(folder + "*/*/*.png")
                for file in raw_file_paths:
                    label = os.path.split(folder)[-1]
                    label = self._find_super_class(label)
                    if label in excluded:
                        continue
                    files.append((self.load_image(file, self.preload_dataset), label))
                    self.all_labels.append(label)
                    if label not in self.unique_labels:
                        self.unique_labels.append(label)

        random.seed(CONFIG.random_seed)
        random.shuffle(files)
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
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_train_dataset, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_validation_dataset, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_test_dataset, pin_memory=True)

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
