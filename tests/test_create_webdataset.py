import io
import os
import tarfile

import numpy as np
import pytest
from PIL import Image

from tests.helpers.create_mock_webdataset_data import create_webdataset_data


@pytest.fixture(params=[True, False])
def unsupervised(request):
    return request.param


@pytest.fixture(params=["png", "jpg"])
def extension(request):
    return request.param


def test_create_webdataset(tmp_path, unsupervised, extension):
    # Create a WebDataset with random noise filled images:
    webdataset_location, image_folder_location = create_webdataset_data(
        path=tmp_path, unsupervised=unsupervised, extension=extension
    )
    assert os.path.exists(webdataset_location)
    assert os.path.isdir(webdataset_location)
    assert len(os.listdir(webdataset_location)) > 3

    tar_files = os.listdir(webdataset_location)
    # check the first and last tarfile:
    for file_path in tar_files[:: len(tar_files) - 1]:
        with tarfile.open(os.path.join(webdataset_location, file_path), "r:*") as opened_file:
            assert len(opened_file.getnames()) > 0
            images = []
            labels = []
            for name in opened_file.getnames():
                _ = images.append(name) if name.endswith(extension) else None
                if not unsupervised:
                    _ = labels.append(name) if name.endswith("txt") else None
                print(name)

            # read a test image from the tar_file:
            tar_info_image = opened_file.getmember(images[0])
            test_image = opened_file.extractfile(tar_info_image)
            test_image = test_image.read()
            test_image = Image.open(io.BytesIO(test_image))
            test_image_array = np.array(test_image)

            # test that the image has the right shape:
            assert len(test_image_array.shape) == 3
            assert test_image_array.shape[2] == 3

            if not unsupervised:
                # read a test label from the tar_file:
                tar_info_label = opened_file.getmember(name=labels[0])
                test_label = opened_file.extractfile(tar_info_label)
                test_label = test_label.read().decode()

                # check that the plain text label is the same as the original folder it was in:
                assert test_label in os.listdir(image_folder_location)

        assert len(images) > 0
        if not unsupervised:
            assert len(images) == len(labels)
