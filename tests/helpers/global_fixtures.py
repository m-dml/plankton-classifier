import pytest
import torch

from tests.helpers.create_mock_classic_dataset import create_classic_dataset_data
from tests.helpers.create_mock_webdataset_data import create_webdataset_data


@pytest.fixture(params=["lars", "sgd", "adam", "rmsprop"], scope="module")
def optimizer(request):
    return request.param


@pytest.fixture(params=["reduce_lr_on_plateau", "linear_warmup_decay", "cosine", None], scope="module")
def scheduler(request):
    return request.param


@pytest.fixture(params=["PytorchProfiler", "NoProfiler"], scope="module")
def profiler(request):
    return request.param


# only create the webdataset once:
@pytest.fixture(scope="session")
def get_webdataset(tmp_path_factory):
    temp_directory = tmp_path_factory.mktemp("webdataset")
    return create_webdataset_data(path=temp_directory, unsupervised=True, extension="png")


@pytest.fixture(scope="session")
def get_image_data_dataset(tmp_path_factory):
    temp_directory = tmp_path_factory.mktemp("image_data")
    return create_classic_dataset_data(path=temp_directory, extension="png")


@pytest.fixture(scope="session")
def cuda_is_available():
    return torch.cuda.is_available()
