import unittest

from src.utils import CONFIG
from src.utils.DataLoader import PlanktonDataLoader


class TestPlanktonDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        config_file = "../tobis_config.yaml"
        CONFIG.update(config_file)

    def test_dataloader_new_data(self):
        new_config = dict(use_klas_data=True,
                          use_planktonnet_data=False,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()

    def test_dataloader_on_planktonnet(self):
        new_config = dict(use_klas_data=False,
                          use_planktonnet_data=True,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()