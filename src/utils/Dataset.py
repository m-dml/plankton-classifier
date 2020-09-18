from torch.utils.data import Dataset

from PIL import Image, ImageOps
import glob
import os
import numpy as np
from tqdm import tqdm
import torch


class PlanktonDataset(Dataset):

    def __init__(self, data_path, transform=None, final_image_size=500):
        """
        Initialization of the Dataset.
        Args:
            data_path (str): Path to the data where the folders with the classes are.
        """
        self.data_path = data_path
        self.final_image_size = final_image_size
        self.transform = transform

        self.images, self.labels = self._load_images_into_memory()

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]

        # if self.transform:
        #     image = self.transform(image)

        return torch.from_numpy(image).float(), torch.from_numpy(np.array(label)).long()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def _get_class_labels(self) -> list:
        """
        Method to generate class labels from folder names.
        Returns:
            (list): List containing all the class labels.
        """
        class_folder_names = glob.glob(os.path.join(self.data_path, "*"))
        if len(class_folder_names) < 1:
            raise FileNotFoundError(f"Did not find any folders with class names at: {self.data_path}")
        class_names = [os.path.split(folder)[-1] for folder in class_folder_names]

        return class_names

    def _count_all_images(self, class_names):
        counter = 0
        for class_name in class_names:
            path_to_images_of_class = os.path.join(self.data_path, class_name)
            images_of_class = glob.glob(os.path.join(path_to_images_of_class, "*.png"))
            for _ in images_of_class:
                counter += 1
        return counter

    def _load_images_into_memory(self):
        class_names = self._get_class_labels()
        n_images = self._count_all_images(class_names=class_names)

        image_array = np.empty([n_images, self.final_image_size, self.final_image_size, 3])
        label_array = np.empty([n_images]).astype(int)
        counter = 0

        for c, class_name in enumerate(tqdm(class_names, desc="loading")):
            path_to_images_of_class = os.path.join(self.data_path, class_name)
            images_of_class = glob.glob(os.path.join(path_to_images_of_class, "*.png"))

            for i, image_file in enumerate(images_of_class):
                image = Image.open(image_file)
                image = ImageOps.pad(image, size=(self.final_image_size, self.final_image_size))
                image_as_array = np.array(image)
                image_array[counter] = image_as_array
                label_array[counter] = c

                counter += 1

        image_array = image_array / 255
        image_array = np.moveaxis(image_array, -1, 1)

        print("Image array shape:", image_array.shape)

        return image_array, label_array
