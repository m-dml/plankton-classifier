import unittest

from planktonclf.utils import CONFIG
from planktonclf.utils.DataLoader import PlanktonDataLoader


class TestPlanktonDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        config_file = "../tobis_config.yaml"
        CONFIG.update(config_file)

    def test_dataset(self):
        pass

    def test_dataloader_old_data_group_classes(self):
        new_config = dict(use_subclasses=False,
                          use_old_data=True,
                          use_new_data=False,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()

    def test_dataloader_old_data_subclasses(self):
        new_config = dict(use_subclasses=True,
                          use_old_data=True,
                          use_new_data=False,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()

    def test_dataloader_new_data(self):
        new_config = dict(use_subclasses=True,
                          use_old_data=False,
                          use_new_data=True,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()