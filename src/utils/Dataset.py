from torch.utils.data import Dataset

from PIL import Image
import glob
import os
import numpy as np


class PlanktonDataset(Dataset):

    def __init__(self, data_path):
        """
        Initialization of the Dataset.

        Args:
            data_path (str): Path to the data where the folders with the classes are.
        """
        self.data_path = data_path
        self.data = None

    def __getitem__(self, item):
        pass

    def __len__(self) -> int:
        pass

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

    def _load_images_into_memory(self):
        class_names = self._get_class_labels()

        for c, class_name in enumerate(class_names):
            path_to_images_of_class = os.path.join(self.data_path, class_name)
            images_of_class = glob.glob(os.path.join(path_to_images_of_class, "*.png"))

            # create empty array to store image_data in by opening the first image and getting its shape:
            image = Image.open(images_of_class[0])
            image_as_array = np.array(image.getdata())
            image_array = np.empty(len(images_of_class), *image_as_array.shape)
            label_array = np.empty(len(images_of_class)).astype(int)

            for i, image_file in enumerate(images_of_class):
                image = Image.open(image_file)
                image_as_array = np.array(image.getdata())
                image_array[i] = image_as_array
                label_array[i] = c
