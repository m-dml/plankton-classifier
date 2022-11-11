import glob
import os

import hydra
from omegaconf import DictConfig

from src.lib.config import register_configs
from src.utils import utils
from tests.helpers.create_and_run_model import create_and_run
from tests.helpers.global_fixtures import get_image_data_dataset, optimizer, profiler, scheduler  # noqa


def test_pretrain_with_folder_dataset(
    tmp_path, get_image_data_dataset, base_optimizer=None, base_scheduler=None, base_profiler=None
):

    base_optimizer = base_optimizer or "lars"
    base_profiler = base_profiler or "NoProfiler"

    accelerator = "cpu"
    devices = 1

    image_location = get_image_data_dataset

    out_path = os.path.join(tmp_path, "output")
    os.makedirs(out_path, exist_ok=True)
    os.chdir(out_path)

    overrides = [
        "+experiment=unit_tests/test_pretrain_with_folder_dataset",
        f"datamodule.data_base_path={image_location}",
        f"output_dir_base_path={out_path}",
        f"optimizer={base_optimizer}",
        f"+profiler={base_profiler}",
        f"trainer.accelerator={accelerator}",
        f"trainer.devices={devices}",
    ]

    if base_scheduler:
        overrides.append(f"+scheduler={base_scheduler}")

    register_configs()
    with hydra.initialize(config_path="../conf", job_name="unit_test_pretrain_image_dataset"):
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
        assert len(glob.glob(os.path.join(out_path, "logs", "checkpoints", "*ckpt"))) > 0
        assert os.path.exists(os.path.join(out_path, "logs", "checkpoints", "last.ckpt"))

    return out_path


# the following functions make sure we train not every option with every other option, but only each choice once:
def test_pretrain_with_image_folder_optimizers(tmp_path, get_image_data_dataset, optimizer):
    test_pretrain_with_folder_dataset(tmp_path, get_image_data_dataset, base_optimizer=optimizer)


def test_pretrain_with_image_folder_schedulers(tmp_path, get_image_data_dataset, scheduler):
    test_pretrain_with_folder_dataset(tmp_path, get_image_data_dataset, base_scheduler=scheduler)


def test_pretrain_with_image_folder_profilers(tmp_path, get_image_data_dataset, profiler):
    test_pretrain_with_folder_dataset(tmp_path, get_image_data_dataset, base_profiler=profiler)
