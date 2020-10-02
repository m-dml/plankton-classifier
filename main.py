from torchvision import transforms
import pytorch_lightning as pl
from src.utils.DataModule import PlanktonDataModule
from src.utils.Model import PlanktonCLF
from argparse import ArgumentParser
import logging
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import GPUStatsMonitor


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

    datamodule = datamodule_class.from_argparse_args(args, transform=transform,
                                                     num_workers=20
                                                     )
    datamodule.setup()

    model = PlanktonCLF(num_classes=len(datamodule.labels), labels=datamodule.labels, **vars(args))

    callbacks = [GPUStatsMonitor()]

    comet_logger = pl_loggers.CometLogger(save_dir="comet_logs", experiment_name="testing-comet-logger",
                                          project_name="plankton-classifier", offline=True)
    trainer = pl.Trainer.from_argparse_args(args, progress_bar_refresh_rate=10, gpus=1,
                                            max_epochs=100,
                                            # distributed_backend='ddp', num_nodes=1,
                                            logger=comet_logger,
                                            callbacks=callbacks
                                            )
    trainer.fit(model, datamodule)
    trainer.test(model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format='%(name)s %(levelname)s %(message)s')
    main()
