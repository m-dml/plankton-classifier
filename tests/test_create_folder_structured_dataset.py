import os

import numpy as np
import pytest
from PIL import Image

from tests.helpers.create_mock_classic_dataset import create_classic_dataset_data


@pytest.fixture(params=["png", "jpg"])
def extension(request):
    return request.param


def test_create_folder_structured_dataset(tmp_path, extension):
    # Create a WebDataset with random noise filled images:
    image_folder_location = create_classic_dataset_data(path=tmp_path, extension=extension)
    assert os.path.exists(image_folder_location)
    assert os.path.isdir(image_folder_location)
    assert len(os.listdir(image_folder_location)) > 1

    image_files_paths = os.listdir(image_folder_location)
    # check the first and last image:

    for label in image_files_paths:
        images = []
        for image_file in os.listdir(os.path.join(image_folder_location, label)):
            test_image = Image.open(os.path.join(image_folder_location, label, image_file))
            test_image_array = np.array(test_image)

            # test that the image has the right shape:
            assert len(test_image_array.shape) == 3
            assert test_image_array.shape[2] == 3
            images.append(image_file)

        assert len(images) > 0
