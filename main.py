from torchvision import transforms
import pytorch_lightning as pl
from src.utils.DataModule import PlanktonDataModule
from src.utils.Model import PlanktonCLF
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
import logging
import matplotlib.pyplot as plt
import sklearn


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

    datamodule = datamodule_class.from_argparse_args(args, transform=transform, num_workers=8)
    model = PlanktonCLF(num_classes=7, **vars(args))

    # callbacks = ([pl.callbacks.gpu_stats_monitor])

    trainer = pl.Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, gpus=1,
                                            max_epochs=1,
                                            # distributed_backend='ddp', num_nodes=3,
                                            # callbacks=[CFCallback(example_images=datamodule.val[0][0],
                                            #                       example_labels=datamodule.val[0][1],
                                            #                       class_names=[str(x) for x in range(7)])]
                                            )
    trainer.fit(model, datamodule)
    trainer.test(model)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


class CFCallback(pl.Callback):

    def __init__(self, example_images, example_labels, class_names):
        self.example_images = example_images
        self.example_labels = example_labels
        self.class_names = class_names

    def on_validation_epoch_start(self, trainer, pl_module):
        dim = (self.num_samples, pl_module.hparams.latent_dim)
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            labels = pl_module(self.example_images)
            pl_module.train()

        cm = sklearn.metrics.confusion_matrix(self.example_labels, labels)

        trainer.logger.experiment.add_figure(plot_confusion_matrix(cm=cm, class_names=self.class_names))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN, format='%(name)s %(levelname)s %(message)s')
    main()
