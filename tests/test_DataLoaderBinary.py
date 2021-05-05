import unittest

from src.utils import CONFIG
from src.utils.DataLoaderBinary import PlanktonDataLoader


class TestPlanktonDataLoaderBinary(unittest.TestCase):

    def setUp(self) -> None:
        config_file = "../tobis_config.yaml"
        CONFIG.update(config_file)

    def test_dataloader_new_data(self):
        new_config = dict(use_subclasses=True,
                          use_old_data=False,
                          use_new_data=True,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()

        self.assertEqual(set(dl.unique_labels), {"Noise", "Signal"})
