import glob
import os
import pathlib
from abc import abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd
import PIL.PngImagePlugin
import pytorch_lightning as pl
import torch
from catalyst.data.sampler import BalanceClassSampler, DistributedSamplerWrapper
from hydra.utils import instantiate
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from src.utils import utils


class ParentDataSet(Dataset):
    def __init__(self, integer_labels, final_image_size=500, transform=None, preload_dataset=False):
        self.files = None
        self.integer_labels = integer_labels

        self.transform = transform
        self.preload_dataset = preload_dataset
        self.final_image_size = final_image_size

        self.console_logger = utils.get_logger(__name__)

    def set_files(self, files):
        self.files = files

    def get_labels(self):  # this is used by the torchsampler
        _, label_names = zip(*self.files)
        labels = [self.integer_labels[label_name] for label_name in label_names]
        return labels

    @staticmethod
    def load_file(file):
        this_image = Image.open(file)
        return this_image

    def __len__(self):
        return len(self.files)

    @abstractmethod
    def __getitem__(self, index) -> (Any, Union[torch.Tensor, list], str):
        del index  # for pylint and vulture compliance
        return Any, Union[torch.Tensor, list], str


class PlanktonDataSet(ParentDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class PlanktonInferenceDataSet(ParentDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, _ = self.files[item]
        image_name = str(image)
        if not self.preload_dataset:
            image = self.load_file(image)

        if self.transform:
            image = self.transform(image)

        if isinstance(image, PIL.PngImagePlugin.PngImageFile):
            image = transforms.ToTensor()(image)

        return image, image_name


class PlanktonMultiLabelDataSet(ParentDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        super().__init__(*args, **kwargs)
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
        data_base_path,
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

        self.data_base_path = data_base_path
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

        self.is_set_up = False
        self.console_logger.info("Successfully initialised the datamodule.")

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

        if len(train_subset) == 0:
            if self.use_klas_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.klas_data_path)}")
            if self.use_canadian_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.canadian_data_path)}")
            if self.use_planktonnet_data:
                raise FileNotFoundError(f"Did not find any files under {os.path.abspath(self.planktonnet_data_path)}")

        if len(valid_subset) == 0:
            valid_idx_start = int(len(train_subset) * self.train_split)
            valid_idx_end = int(len(train_subset) * (self.train_split + self.validation_split))

            valid_subset = train_subset[valid_idx_start:valid_idx_end]
            test_subset = train_subset[valid_idx_end:]
            train_subset = train_subset[:valid_idx_start]

        self.console_logger.debug("Separating labels from images")
        self.unique_labels, self.train_labels = np.unique(list(list(zip(*train_subset))[1]), return_inverse=True)
        self.console_logger.debug(f"There are {len(self.unique_labels)} unique training labels: {self.unique_labels}")
        unique_val_labels, self.valid_labels = np.unique(list(list(zip(*valid_subset))[1]), return_inverse=True)
        self.console_logger.debug(f"There are {len(unique_val_labels)} unique validation labels: {unique_val_labels}")
        unique_test_labels, self.test_labels = np.unique(list(list(zip(*test_subset))[1]), return_inverse=True)
        self.console_logger.debug(f"There are {len(unique_test_labels)} unique test labels: {unique_test_labels}")

        self.integer_class_label_dict = self.set_up_integer_class_labels()

        # self._test_label_consistency(self.unique_labels, unique_val_labels)
        # self._test_label_consistency(self.unique_labels, unique_test_labels)

        if self.subsample_supervised <= 1:
            label_dict = {
                label: np.arange(len(self.train_labels))[self.train_labels == label].tolist()
                for label in np.arange(0, len(self.unique_labels.flatten()))
            }

            indices = []
            for key in sorted(label_dict):
                num_samples_this_label = int(np.ceil(len(label_dict[key]) * self.subsample_supervised))
                self.console_logger.debug(
                    f"For class {self.unique_labels[key]} will be using {num_samples_this_label} samples for training "
                    f"from originally {len(label_dict[key])} labeled images. "
                )
                indices += np.random.choice(label_dict[key], num_samples_this_label, replace=False).tolist()
            train_subset = [train_subset[i] for i in indices]
            self.train_labels = self.train_labels[indices]

        self.console_logger.debug("Getting the image counts for each label")
        _, self.training_class_counts = np.unique(self.train_labels, return_counts=True)
        self.len_train_data = int(len(train_subset) / self.batch_size)

        self.console_logger.info(f"There are {len(train_subset)} training files")
        self.console_logger.info(f"There are {len(valid_subset)} validation files")
        if stage == "fit":
            self.console_logger.info(f"Instantiating training dataset <{self.cfg_dataset._target_}>")
            self.train_data: ParentDataSet = instantiate(
                self.cfg_dataset,
                _convert_="all",
                _recursive_=False,
                integer_labels=self.integer_class_label_dict,
                transform=self.train_transforms,
                preload_dataset=self.preload_dataset,
            )

            self.train_data.set_files(train_subset)

            self.console_logger.info(f"Instantiating validation dataset <{self.cfg_dataset._target_}>")
            self.console_logger.debug(f"Number of training samples: {len(self.train_data)}")
            self.valid_data: ParentDataSet = instantiate(
                self.cfg_dataset,
                _convert_="all",
                _recursive_=False,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )

            self.valid_data.set_files(valid_subset)

            self.console_logger.debug(f"Number of validation samples: {len(self.valid_data)}")

        elif stage == "test":
            self.console_logger.info(f"Instantiating test dataset <{self.cfg_dataset._target_}>")
            self.test_data: ParentDataSet = instantiate(
                self.cfg_dataset,
                _convert_="all",
                _recursive_=False,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )
            self.test_data.set_files(test_subset)

        else:
            raise ValueError(f'<stage> needs to be either "fit" or "test", but is {stage}')

        self.is_set_up = True

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
            if os.path.getsize(file) <= 30:  # smallest image(tiff) is 35 bytes (single black pixel png is 67 bytes)
                continue  # skip empty files
            label = os.path.split(folder)[-1]
            label = self._find_super_class(label)
            if self.excluded_labels is not None:
                if label in self.excluded_labels:
                    continue
            files.append((self.load_image(file, self.preload_dataset), label))
            self.all_labels.append(label)

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
            if self.subsample_supervised <= 1:
                subsamples = "upsampling"
            else:
                subsamples = int(self.subsample_supervised)
            sampler = BalanceClassSampler(self.train_labels, mode=subsamples)
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


class PlanktonInferenceDataLoader(PlanktonDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(
            **kwargs,
        )

    def setup(self, *args, **kwargs):
        if isinstance(self.unlabeled_files_to_append, str):
            self.unlabeled_files_to_append = [self.unlabeled_files_to_append]
        if self.unlabeled_files_to_append is None:
            raise ValueError(
                "You have to provide a folder of images for inference sessions. Use "
                "`datamodule.unlabeled_files_to_append=/path/to/folder` when calling the script"
            )

        for filepath in self.unlabeled_files_to_append:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"The provided folder does not exist: <{filepath}>")

        predict_subset = []

        for folder_with_unlabeled_files in self.unlabeled_files_to_append:
            predict_subset += self.add_all_images_from_all_subdirectories(folder_with_unlabeled_files)

        if len(predict_subset) == 0:
            raise FileNotFoundError(f"Did not find any files under {self.unlabeled_files_to_append}")

        self.console_logger.debug("Getting the image counts for each label")
        self.len_train_data = int(len(predict_subset) / self.batch_size)

        self.console_logger.info(f"There are {len(predict_subset)} inference files")

        self.console_logger.info(f"Instantiating test dataset <{self.cfg_dataset._target_}>")

        self.test_data: ParentDataSet = instantiate(
            self.cfg_dataset,
            _convert_="all",
            _recursive_=False,
            integer_labels=self.integer_class_label_dict,
            transform=self.valid_transforms,
            preload_dataset=self.preload_dataset,
        )
        self.test_data.set_files(predict_subset)

        # the train dataset is needed only for setup but will not be used for inference:
        self.train_data: ParentDataSet = instantiate(
            self.cfg_dataset,
            _convert_="all",
            _recursive_=False,
            integer_labels=self.integer_class_label_dict,
            transform=self.valid_transforms,
            preload_dataset=self.preload_dataset,
        )
        self.train_data.set_files(predict_subset)

        self.valid_data: ParentDataSet = instantiate(
            self.cfg_dataset,
            _convert_="all",
            _recursive_=False,
            integer_labels=self.integer_class_label_dict,
            transform=self.valid_transforms,
            preload_dataset=self.preload_dataset,
        )
        self.valid_data.set_files(predict_subset)

        self.is_set_up = True


class PlanktonMultiLabelDataLoader(PlanktonDataLoader):
    def __init__(
        self,
        csv_data_path,
        convert_probabilities_to_majority_vote,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.label_encoder = None
        self.csv_data_path = csv_data_path
        self.unique_labels = None
        self.convert_probabilities_to_majority_vote = convert_probabilities_to_majority_vote

    def setup(self, stage=None):
        train_subset = self.prepare_data_setup("train")
        valid_subset = self.prepare_data_setup("val")
        test_subset = self.prepare_data_setup("test")

        if self.oversample_data:
            if self.subsample_supervised <= 1:
                train_indices = np.arange(0, len(train_subset))
                train_subset = [
                    train_subset[x]
                    for x in np.random.choice(
                        train_indices, int(self.subsample_supervised * len(train_subset)), replace=False
                    ).tolist()
                ]
            else:
                raise NotImplementedError(
                    "For Multilabling just percentage subsampling is possible, so subsample_supervised has to be <= 1 ."
                )

        self.len_train_data = int(len(train_subset) / self.batch_size)

        self.train_labels = list(zip(*train_subset))[1]
        self.valid_labels = list(zip(*valid_subset))[1]
        self.test_labels = list(zip(*test_subset))[1]

        self.console_logger.info(f"There are {len(train_subset)} training files")
        self.console_logger.info(f"There are {len(valid_subset)} validation files")
        if stage == "fit":
            self.console_logger.info(f"Instantiating training dataset <{self.cfg_dataset._target_}>")
            self.train_data: ParentDataSet = instantiate(
                self.cfg_dataset,
                integer_labels=self.train_labels,
                transform=self.train_transforms,
                preload_dataset=self.preload_dataset,
            )
            self.train_data.set_files(train_subset)

            self.console_logger.info(f"Instantiating validation dataset <{self.cfg_dataset._target_}>")
            self.valid_data: ParentDataSet = instantiate(
                self.cfg_dataset,
                integer_labels=self.valid_labels,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )
            self.valid_data.set_files(valid_subset)

            self.console_logger.debug(f"Number of validation samples: {len(self.valid_data)}")

        elif stage == "test":
            self.console_logger.info(f"Instantiating test dataset <{self.cfg_dataset._target_}>")
            self.test_data: ParentDataSet = instantiate(
                self.cfg_dataset,
                integer_labels=self.integer_class_label_dict,
                transform=self.valid_transforms,
                preload_dataset=self.preload_dataset,
            )
            self.test_data.set_files(test_subset)

        else:
            raise ValueError(f'<stage> needs to be either "fit" or "test", but is {stage}')

    def load_multilabel_dataset(self, data_path, csv_file):
        df = pd.read_csv(csv_file)
        df = df.drop(columns="Unnamed: 0")
        repl_column_names = dict()
        self.console_logger.debug(f"Created dataframe with {len(df)} rows and columns: {df.columns}")
        nan_vals = df.isna().sum().sum()
        if nan_vals > 0:
            self.console_logger.warning(f"Found {nan_vals} rows with NaN values. I will drop them.")
        df = df.dropna()
        all_labels = []
        for column in df.columns:
            column_new = column.strip().lower()
            column_new = column_new[3:] if column_new.startswith("00_") else column_new
            repl_column_names[column] = column_new

            if column != "file":
                all_labels += df[column].values.tolist()

        if self.label_encoder is None:
            self.label_encoder = preprocessing.LabelEncoder()
            self.label_encoder.fit(all_labels)

        for column in df.columns:
            if column != "file":
                df[column] = self.label_encoder.transform(df[column].values)

        df = df.rename(columns=repl_column_names)

        if self.unique_labels is None:
            self.unique_labels = self.label_encoder.classes_.tolist()
            self.max_label_value = df.drop(labels="file", axis=1).max().max()
        if not set(np.unique(all_labels)).issubset(self.unique_labels):
            new_labels = set(np.unique(all_labels)).difference(set(self.unique_labels))
            raise ValueError(f"The labels {new_labels} from <{csv_file}> are not in the list of training labels.")
        if (set(np.unique(all_labels)) != set(self.unique_labels)) and set(np.unique(all_labels)).issubset(
            self.unique_labels
        ):
            self.console_logger.warning(
                f"There are labels in the training dataset that are not in <{csv_file}> dataset."
            )

        self.console_logger.debug(f"All unique labels from labelencoder are {self.unique_labels}")
        self.console_logger.debug(f"All unique labels from np.unique are {np.unique(all_labels)}")

        files = []

        for file, labels in df.set_index("file").iterrows():
            files.append(
                (
                    self.load_image(os.path.join(data_path, file), preload=self.preload_dataset),
                    self.multi_labels_to_probabilities(labels.values),
                    labels.values,
                )
            )
        return files

    def prepare_data_setup(self, subset):
        file_struct = os.path.join(self.csv_data_path, f"*{subset}*.csv")
        self.console_logger.info(f"Trying to load csv file from <{file_struct}>")
        csv_file = glob.glob(file_struct)[0]
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(
                f"Could not find csv file <{csv_file}> in " f"<{os.path.join(self.csv_data_path, f'*{subset}*.csv')}>"
            )
        self.console_logger.info(f"Loading data from <{self.csv_data_path} ; {csv_file}>")
        folder = os.path.join(self.csv_data_path, subset)
        if not os.path.isdir(folder):
            self.console_logger.warning(f"Could not find folder <{folder}>. Using <{self.data_base_path}> instead.")
            folder = self.data_base_path
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Could not find <{folder}>")
        self.console_logger.info(f"Reading relative paths in csv file starting from <{folder}>")
        files = self.load_multilabel_dataset(folder, csv_file)
        return files

    def multi_labels_to_probabilities(self, labels):
        n_bins = len(self.unique_labels)
        probabilities = np.histogram(labels, bins=n_bins, range=(0, n_bins))[0] / len(labels)

        if self.convert_probabilities_to_majority_vote:
            probabilities = np.argmax(probabilities)
        return probabilities

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle_train_dataset,
        )


class PlanktonMultiLabelSingleScientistDataLoader(PlanktonDataLoader):
    def __init__(
        self,
        csv_data_path,
        which_expert_label: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.csv_data_path = csv_data_path
        self.which_expert_label = which_expert_label

    def prepare_data_setup(self, subset):
        return PlanktonMultiLabelDataLoader.prepare_data_setup(self, subset)

    def load_multilabel_dataset(self, data_path, csv_file):
        df = pd.read_csv(csv_file)
        df = df.drop(columns="Unnamed: 0")
        repl_column_names = dict()
        for column in df.columns:
            column_new = column.strip().lower()
            column_new = column_new[3:] if column_new.startswith("00_") else column_new
            repl_column_names[column] = column_new

        df = df.rename(columns=repl_column_names)

        files = []
        self.unique_labels = np.arange(0, self.max_label_value + 1)
        for file, labels in df.set_index("file").iterrows():
            files.append(
                (
                    self.load_image(os.path.join(data_path, file), preload=self.preload_dataset),
                    self.choose_label_from_scientist(labels.values),
                )
            )
        return files

    def choose_label_from_scientist(self, label_list):
        return label_list[self.which_expert_label]
