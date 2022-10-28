from abc import abstractmethod

import hydra.utils
import pytorch_lightning as pl
import torch
import webdataset


class BaseWebDataset(webdataset.WebDataset):
    def __init__(self, integer_labels, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.integer_labels = integer_labels
        self.transform = transform


class FinetuneWebDataset(BaseWebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        image, label_name = item["input"], item["label"]

        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        return image, (torch.Tensor([self.integer_labels[label_name]]), label_name)


class PretrainWebDataset(BaseWebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        image = item["input"]
        image_copy = image.copy()

        if self.transform:
            image = self.transform(image)
            image_copy = self.transform(image_copy)
        else:
            raise ValueError("Transforms must be defined for pretraining")

        return (image, image_copy), (torch.tensor([0]))


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
        super_classes,
        oversample_data,
        random_seed,
        train_transforms,
        valid_transforms,
        data_base_path,
        dataset,
        is_in_simclr_mode,
        reduce_data,
        pin_memory=False,
        unlabeled_files_to_append=None,
        is_ddp=False,
        subsample_supervised=100,
        shuffle_size=5000,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.excluded_labels = excluded_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train_dataset = shuffle_train_dataset
        self.shuffle_validation_dataset = shuffle_validation_dataset
        self.shuffle_size = shuffle_size
        self.cfg_dataset = dataset
        self.data_base_path = data_base_path
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    def make_loader(self, paths, mode="fit"):
        shuffle = 0
        if mode == "fit":
            transforms = hydra.utils.instantiate(self.train_transforms)
            if isinstance(self.shuffle_train_dataset, bool):
                shuffle = self.shuffle_size if self.shuffle_train_dataset else 0

        elif mode == "eval":
            transforms = hydra.utils.instantiate(self.valid_transforms)
            if isinstance(self.shuffle_validation_dataset, bool):
                shuffle = self.shuffle_size if self.shuffle_validation_dataset else 0
        else:
            raise ValueError("Mode must be either 'fit' or 'eval'")

        dataset = (
            hydra.utils.instantiate(self.cfg_dataset)
            .shuffle(shuffle)
            .decode("pil")
            .tu_tuple()
            .map_tuple(transforms, lambda x: x)  # lamda for not transforming the labels (identity func)
            .batched(self.batch_size, partial=False)
        )

        loader = webdataset.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=self.num_workers)

        if mode is "fit":
            loader = loader.ddp_equalize(dataset_size // self.batch_size)
