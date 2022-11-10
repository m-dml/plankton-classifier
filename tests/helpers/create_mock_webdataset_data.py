import os
from typing import Union

from tests.helpers.create_mock_classic_dataset import create_classic_dataset_data
from wds_scripts.create_webdataset import create_unsupervised_dataset_from_folder_structure


def create_webdataset_data(
    path: Union[str, os.PathLike],
    num_images_per_subdir: int = 100,
    num_channels: int = 3,
    max_width: int = 24,
    max_height: int = 24,
    random_sizes: bool = True,
    unsupervised: bool = True,
    shard_size: int = 1e4,
    extension: str = "png",
) -> (str, str):
    """Creates a WebDataset with random noise filled images.

    Args:
        path (str): Base path where the folders "webdataset" and "images" should be created.
        num_images_per_subdir (int, optional): Number of images per subdirectory. 4 subdirectories will be created. The
            subdirectories are used to create names for supervised learning. Defaults to 100.
        num_channels (int, optional): Number of channels for the images. Defaults to 3.
        max_width (int, optional): Maximum width of the images. If random_size is False, then this will be the exact
            width of all images. Defaults to 24.
        max_height (int, optional): Maximum height of the images. If random_size is False, then this will be the exact
            height of all images. Defaults to 24.
        random_sizes (bool, optional): If True, then the images will have random sizes. If False, then all images will
            have the same size. Defaults to True.
        unsupervised (bool, optional): If True, then the webdataset will not contain any labels.
        shard_size (int, optional): Size of the shards. Defaults to 1e4.
        extension (str, optional): Extension (output format) of the images. Defaults to "png".

    Returns:
        (str, str): Path to the webdataset and path to the images.
    """
    webdataset_location = os.path.join(path, "webdataset")

    os.makedirs(webdataset_location, exist_ok=True)

    # create and save images:
    image_location = create_classic_dataset_data(
        path=path,
        num_images_per_subdir=num_images_per_subdir,
        num_channels=num_channels,
        max_width=max_width,
        max_height=max_height,
        random_sizes=random_sizes,
        extension=extension,
    )

    # use the saved images to create a webdataset:
    create_unsupervised_dataset_from_folder_structure(
        src_path=image_location,
        dst_path=webdataset_location,
        dst_prefix="",
        shard_size=shard_size,
        unsupervised=unsupervised,
        extension=extension,
    )
    return webdataset_location, image_location
