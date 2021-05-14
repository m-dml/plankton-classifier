import unittest

from src.utils import CONFIG
from src.utils.DataLoader import PlanktonDataLoader


class TestPlanktonDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        config_file = "../tobis_config.yaml"
        CONFIG.update(config_file)

    def test_dataloader_klas_data(self):
        new_config = dict(use_klas_data=True,
                          use_planktonnet_data=False,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()

        self._dataloader_tests(dl)

    def test_dataloader_on_planktonnet(self):
        new_config = dict(use_klas_data=False,
                          use_planktonnet_data=True,
                          preload_dataset=False
                          )
        CONFIG.update(new_config)
        dl = PlanktonDataLoader()
        dl.setup()

        self._dataloader_tests(dl)

    def _dataloader_tests(self, dl):
        self.assertIsNotNone(dl.integer_class_label_dict)
        for integer_label in dl.integer_class_label_dict.values():
            self.assertIsInstance(integer_label, int)

        self.assertFalse([] == dl.unique_labels)
        self.assertIsNotNone(dl.train_data)
        self.assertIsNotNone(dl.valid_data)
        self.assertIsNotNone(dl.test_data)