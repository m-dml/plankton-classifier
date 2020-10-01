from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from .Dataset import PlanktonDataset
from argparse import ArgumentParser


class PlanktonDataModule(LightningDataModule):

    def __init__(self, data_path='data/plankton_dataset/Training3_0', final_image_size=500, train_test_split=0.9,
                 train_valid_split=0.9, batch_size=8, transform=None, data_is_grouped=False, **kwargs):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.final_image_size = final_image_size
        self.train_test_split = train_test_split
        self.train_valid_split = train_valid_split
        self.transform = transform
        self.data_is_grouped = data_is_grouped

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage=None):
        # called on every GPU

        dataset = PlanktonDataset(data_path=self.data_path, transform=self.transform,
                                  final_image_size=self.final_image_size, data_is_grouped=self.data_is_grouped)
        train_valid_split = int(len(dataset) * self.train_test_split)
        test_split = len(dataset) - train_valid_split
        train_valid_dataset, test_dataset = random_split(dataset, [train_valid_split, test_split])
        if stage == 'fit' or stage is None:
            train_split = int(len(train_valid_dataset) * self.train_valid_split)
            valid_split = len(train_valid_dataset) - train_split
            self.train, self.val = random_split(train_valid_dataset, [train_split, valid_split])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)

    @staticmethod
    def add_argparse_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='data/plankton_dataset/VPR_M87_grouped')
        parser.add_argument('--data_is_grouped', action="store_true", default=False)
        return parser
