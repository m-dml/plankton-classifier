"""Module containing pytorch models and functions only applied to pytorch models.

Classes:
1. Classifier: A simple classifier with a variable number of fully connected hidden layers.
2. CustomResnet: A custom resnet where the kernel size of the first convolutional layer can be changed.

Functions:
concat_feature_extractor_and_classifier: Concatenates a feature extractor and a classifier into a single model.

"""

import warnings
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from src.utils import utils


class Classifier(nn.Module):
    """Classifier with a variable number of fully connected hidden layers and batch normalization."""

    def __init__(
        self,
        num_classes: int,
        hidden_layers: Iterable = (1000, 1000),
        activation=nn.ReLU(),
        input_features: int = 1000,
        normalize: bool = False,
        bias_in_last_layer: bool = True,
    ):
        """Initialize the classifier.

        Even if an empty list for hidden_layers is provided, one output layer without activation will be added.

        Args:
            num_classes (int): Number of classes.
            hidden_layers (Iterable): Iterable containing the number of nodes in each hidden layer. len(hidden_layer)
                therefore determines the number of hidden layers. If an empty iterable is passed, no hidden layers are
                used. Automatically, no matter hwo many hidden layers are used, a final output layer without activation
                is added.
            activation (nn.Module): Activation function to be used between the hidden layers.
            input_features (int): Number of input features. If the classifier is used together with a feature extractor,
                this should be the number of output features of the feature extractor.
            normalize (bool): If True, the output of the classifier is normalized using torch.nn.functional.normalize().
            bias_in_last_layer (bool): If True, the last layer has a bias. If False, the last layer has no bias.
                Default is True.

        """
        super().__init__()
        self.normalize = normalize
        self.hidden_layers = list(hidden_layers)
        self.num_classes = num_classes
        self.activation = activation
        self.hidden_layers.insert(0, input_features)

        modules = []
        for i in range(len(self.hidden_layers) - 1):
            modules.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
            modules.append(nn.BatchNorm1d(self.hidden_layers[i + 1]))
            modules.append(activation)

        modules.append(nn.Linear(self.hidden_layers[-1], num_classes, bias=bias_in_last_layer))
        self.model = nn.Sequential(*modules)

    def forward(self, input_tensor) -> torch.Tensor:
        """Forward pass of the classifier.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        """
        if self.normalize:
            input_tensor = self.model(input_tensor)
            return F.normalize(input_tensor, dim=1)

        return self.model(input_tensor)


class CustomResnet(nn.Module):
    """Custom implementation of a resnet where the kernel size of the first convolutional layer can be changed and max-
    pooling can be disabled."""

    def __init__(
        self, model: torch.nn.Module, kernel_size: int = 7, stride: int = 2, channels: int = 3, maxpool1: bool = True
    ):
        """
        Args:
            model (torch.nn.Module): Original resnet model.
            kernel_size (int, optional): Kernel size of the first convolutional layer. Defaults to 7.
            stride (int, optional): Stride of the first convolutional layer. Defaults to 2.
            channels (int, optional): Number of channels of the input tensor. Defaults to 3.
            maxpool1 (bool, optional): Whether to use max-pooling after the first convolutional layer. Defaults to True.
        """

        super().__init__()
        self.model = model

        if not maxpool1:
            self.model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        conv1_out_channels = self.model.conv1.out_channels
        self.model.conv1 = nn.Conv2d(
            channels, conv1_out_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the resnet wrapper to the original resnet.

        Args:
            image_tensor (torch.Tensor): Input tensor.

        """
        return self.model(image_tensor)


class TinyFeatureExtractor(nn.Module):
    """Feature extractor for unit tests."""

    def __init__(self, channels: int = 3, n_features: int = 1000):
        """Initialize the feature extractor.

        Args:
            channels (int, optional): Number of channels of the input tensor. Defaults to 3.
            n_features (int, optional): Number of output features. Defaults to 1000.

        """

        super().__init__()
        self.console_logger = utils.get_logger(__name__)
        self.console_logger.warning("Using tiny feature extractor. Make sure this is not used in production.")
        warnings.warn("Using tiny feature extractor. Make sure this is not used in production.", UserWarning)

        self.first_conv = nn.Conv2d(channels, 3, kernel_size=3, stride=2, padding=0, bias=False)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fully_connected = nn.Linear(3 * (1 * 1), n_features)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feature extractor.

        Args:
            image_tensor (torch.Tensor): Input tensor.

        """
        features = F.relu(self.first_conv(image_tensor))
        features = self.average_pool(features)
        features = features.flatten(start_dim=1)
        features = self.fully_connected(features)
        return features
