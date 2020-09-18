from torchvision import datasets, transforms
from torch.utils.data import random_split

from torch.utils.data import DataLoader
import pytorch_lightning
from src.utils.DataModule import PlanktonDataModule
from src.utils.Model import LitModel
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    logger = TensorBoardLogger(save_dir="tb_logs")
    transform = transforms.Compose([transforms.RandomVerticalFlip(), transforms.RandomRotation(degrees=180),
                                    transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.5, 0.5, 0.5),
                                    transforms.ToTensor()])
    datamodule = PlanktonDataModule(data_path='data/plankton_dataset/Training3_0', transform=transform, batch_size=8)

    trainer = pytorch_lightning.Trainer(gpus=1, max_epochs=20, logger=logger)
    model = LitModel()

    trainer.fit(model, datamodule)
