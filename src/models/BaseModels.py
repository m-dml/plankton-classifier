from typing import List

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_layers: List[int] = (1000, 1000),
        activation=nn.ReLU(),
        input_features: int = 1000,
    ):
        super(Classifier, self).__init__()
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.hidden_layers.insert(0, input_features)

        self.model = nn.Sequential(
            *[
                self.activation(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
                for i in range(len(self.hidden_layers) - 1)
            ]
            + [nn.Linear(self.hidden_layers[-1], num_classes)]
        )

    def forward(self, x):
        return self.model(x)


class SimCLRFeatureExtractor(nn.Module):
    def __init__(self, model, device):
        super(SimCLRFeatureExtractor, self).__init__()
        self.model = model
        self.device = device

    def forward(self, image_tuples):
        image_transformations_1, image_transformations_2 = list(zip(*image_tuples))
        features_1 = self.model(torch.tensor(image_transformations_1, device=self.device))
        features_2 = self.model(torch.tensor(image_transformations_2, device=self.device))
        proj_features = torch.cat([features_1, features_2], dim=0)
        return proj_features


def load_state_dict(model, checkpoint):
    return model.load_state_dict(checkpoint, strict=True)


def concat_feature_extractor_and_classifier(feature_extractor, classifier):
    model = nn.Sequential(feature_extractor, classifier)
    return model
