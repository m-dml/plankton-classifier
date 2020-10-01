from torchvision import transforms
import pytorch_lightning as pl
from src.utils.DataModule import PlanktonDataModule
from src.utils.Model import PlanktonCLF
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser


def main(args=None):
    pl.seed_everything(52)

    datamodule_class = PlanktonDataModule

    parser = ArgumentParser()

    script_args, _ = parser.parse_known_args(args)
    parser = datamodule_class.add_argparse_args(parser)
    parser = PlanktonCLF.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    transform = transforms.Compose([transforms.RandomVerticalFlip(), transforms.RandomRotation(degrees=180),
                                    transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(0.2, 0.2, 0.2),
                                    transforms.ToTensor()])

    datamodule = datamodule_class.from_argparse_args(args, transform=transform)
    model = PlanktonCLF(**vars(args))
    model.load_from_checkpoint(checkpoint_path="lightning_logs/version_")

    trainer = pl.Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, gpus=1,
                                            max_epochs=1,
                                            distributed_backend='ddp', num_nodes=3
                                            )
    trainer.test(model, )


if __name__ == '__main__':
    main()
