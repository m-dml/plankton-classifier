import glob
import os

import hydra
import pytest
import torch
from omegaconf import DictConfig

from src.lib.config import register_configs
from tests.helpers.create_and_run_model import create_and_run
from tests.helpers.create_mock_webdataset_data import create_webdataset_data
from src.utils import utils

@pytest.fixture(params=["lars", "sgd", "adam", "rmsprop"])
def optimizer(request):
    return request.param


@pytest.fixture(params=["reduce_lr_on_plateau", "linear_warmup_decay", "cosine", None])
def scheduler(request):
    return request.param


@pytest.fixture(params=["PytorchProfiler", "NoProfiler"])
def profiler(request):
    return request.param


# only create the webdataset once:
@pytest.fixture(scope="session")
def get_webdataset(tmp_path_factory):
    temp_directory = tmp_path_factory.mktemp("webdataset")
    return create_webdataset_data(path=temp_directory, unsupervised=True, extension="png")


@pytest.fixture(scope="session")
def cuda_is_available():
    return torch.cuda.is_available()


def test_pretrain_with_webdataset(tmp_path, optimizer, scheduler, profiler, get_webdataset, cuda_is_available):
    accelerator = "cuda" if cuda_is_available else "cpu"
    devices = 1 if accelerator == "cuda" else int(os.cpu_count() / 2)

    webdataset_location, image_location = get_webdataset

    out_path = os.path.join(tmp_path, "output")
    os.makedirs(out_path, exist_ok=True)
    os.chdir(out_path)

    overrides = [
        "+experiment=unit_tests/test_pretrain_with_webdataset",
        f"datamodule.data_base_path={webdataset_location}",
        f"output_dir_base_path={out_path}",
        f"optimizer={optimizer}",
        f"+profiler={profiler}",
        f"trainer.accelerator={accelerator}",
        f"trainer.devices={devices}",
    ]

    if scheduler:
        overrides.append(f"+scheduler={scheduler}")

    register_configs()
    with hydra.initialize(config_path="../conf", job_name="unit_test_pretrain_webdataset"):
        cfg = hydra.compose(config_name="config", overrides=overrides)

        # Pretty print config using Rich library
        if cfg.print_config:
            utils.print_config(cfg, resolve=True)  # prints the complete hydra config to std-out

        print(f"Current working dir: {os.getcwd()}")

        assert isinstance(cfg, DictConfig)

        # create and run model, including dataloader etc.:
        loss = create_and_run(cfg)

        assert loss is not None
        assert loss > 0
        assert os.path.isdir(os.path.join(out_path, "logs", "checkpoints"))
        assert len(glob.glob(os.path.join(out_path,  "logs", "checkpoints", "*ckpt"))) > 0
        assert os.path.exists(os.path.join(out_path, "logs", "checkpoints", "last.ckpt"))
