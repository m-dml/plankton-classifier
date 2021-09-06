import torch.nn as nn


def ResNet(resnet, num_classes):
    classifier = nn.Linear(1000, num_classes)
    model = nn.Sequential(resnet, classifier)
    return model
