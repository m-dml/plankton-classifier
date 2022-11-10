import os
from typing import Union

import numpy as np
from PIL import Image


def create_image(width: int, height: int, num_channels: int) -> np.ndarray:
    """
    Creates an image with fixed width, height and number of channels. The intensity of each pixel is random.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        num_channels (int): Number of channels of the image.

    Returns:
        np.ndarray: Image with random intensity.
    """
    image = np.random.randint(0, 255, (width, height, num_channels), dtype=np.uint8)
    return image


def create_image_with_random_size(max_width: int, max_height: int, num_channels: int) -> np.ndarray:
    """
    Creates an image with random width, height and fixed number of channels. The intensity of each pixel is random.

    Args:
        max_width (int): Maximum width of the image.
        max_height (int): Maximum height of the image.
        num_channels (int): Number of channels of the image.

    Returns:
        np.ndarray: Image with random intensity.
    """
    width = np.random.randint(1, max_width)
    height = np.random.randint(1, max_height)
    return create_image(width, height, num_channels)


def save_image(image: np.ndarray, path: Union[str, os.PathLike], name: str, extension="png"):
    """
    Saves a numpy array as an image to a given path with a given name and extension.

    Args:
        image (np.ndarray): Image to save. Should be a numpy array with shape (width, height, num_channels).
        path (str): Path where the image should be saved.
        name (str): Basename of the image.
        extension (str, optional): Extension of the image. Defaults to "png".
    """
    image = Image.fromarray(image)
    image.save(os.path.join(path, f"{name}.{extension}"))


def create_and_save_n_images(
    path: Union[str, os.PathLike],
    num_images: int,
    num_channels: int = 3,
    max_width: int = 24,
    max_height: int = 24,
    random_sizes: bool = True,
    extension: str = "png",
) -> str:
    """
    Creates and saves n images to a given path.

    Args:
        path (str): Path where the images should be saved.
        num_images (int): Number of images to create.
        num_channels (int, optional): Number of channels of the images. Defaults to 3.
        max_width (int, optional): Maximum width of the images. Defaults to 24.
        max_height (int, optional): Maximum height of the images. Defaults to 24.
        random_sizes (bool, optional): If True, then the images will have random sizes.
        extension (str, optional): Extension of the images. Defaults to "png".

    Returns:
        str: Folder into which the images were saved.
    """
    for i in range(num_images):
        if random_sizes:
            image = create_image_with_random_size(max_width, max_height, num_channels)
        else:
            image = create_image(max_width, max_height, num_channels)
        save_image(image=image, path=path, name=f"image_{i:07d}", extension=extension)

    return path
