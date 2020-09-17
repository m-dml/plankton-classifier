from torchvision import datasets
from torch.utils.data import random_split

from fastai.data.load import DataLoader
from fastai.vision.data import ImageDataLoaders
import fastai
from fastai.vision.learner import cnn_learner
from fastai.metrics import accuracy_multi

from torchvision.models import densenet121
from fastai.vision.augment import crop_pad

from fastai.vision.models.xresnet import xresnet18


if __name__ == '__main__':

    # the data has to be at plankton-classifier/data/Training3_0 for the next line to work
    dataset = datasets.ImageFolder('data/Training3_0')



    # create a train-test split with 90% training data and 10% testing data
    train_split = int(len(dataset) * 0.9)
    test_split = len(dataset) - train_split
    train_dataset, test_dataset = random_split(dataset, [train_split, test_split])

    data_loader = DataLoader(dataset, shuffle=True, batch_size=32)

    batch_size = 32
    data = ImageDataLoaders.from_folder('data/Training3_0', bs=batch_size)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    learner = cnn_learner(data, densenet121, pretrained=True, metrics=accuracy_multi)
