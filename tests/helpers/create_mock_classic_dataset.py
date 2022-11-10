import os
from typing import Union

from tests.helpers.create_mock_images import create_and_save_n_images


def create_classic_dataset_data(
    path: Union[str, os.PathLike],
    num_images_per_subdir: int = 100,
    num_channels: int = 3,
    max_width: int = 24,
    max_height: int = 24,
    random_sizes: bool = True,
    extension: str = "png",
) -> str:
    """Creates a dataset with the folder names being the labels and with random noise filled images.

    Args:
        path (str): Base path where the folder "images" should be created.
        num_images_per_subdir (int, optional): Number of images per subdirectory. 4 subdirectories will be created. The
            subdirectories are used to create names for supervised learning. Defaults to 100.
        num_channels (int, optional): Number of channels for the images. Defaults to 3.
        max_width (int, optional): Maximum width of the images. If random_size is False, then this will be the exact
            width of all images. Defaults to 24.
        max_height (int, optional): Maximum height of the images. If random_size is False, then this will be the exact
            height of all images. Defaults to 24.
        random_sizes (bool, optional): If True, then the images will have random sizes. If False, then all images will
            have the same size. Defaults to True.
        extension (str, optional): Extension (output format) of the images. Defaults to "png".

    Returns:
        (str, str): Path to the webdataset and path to the images.
    """
    image_location = os.path.join(path, "images")

    os.makedirs(image_location, exist_ok=True)

    # create and save images:
    for folder_name in ["folder_a", "folder_b", "folder_c", "folder_d"]:
        image_subdir = os.path.join(image_location, folder_name)
        os.makedirs(image_subdir, exist_ok=True)
        _ = create_and_save_n_images(
            path=image_subdir,
            num_images=num_images_per_subdir,
            num_channels=num_channels,
            max_width=max_width,
            max_height=max_height,
            random_sizes=random_sizes,
            extension=extension,
        )

    return image_location
