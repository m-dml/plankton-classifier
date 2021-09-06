import hydra
import torch.nn as nn
import torch.nn.functional as F


class ResNet:
    def __init__(self, cfg_resnet, num_classes):

        try:
            feature_extractor = hydra.utils.instantiate(cfg_resnet, aux_logits=False)
        except:
            feature_extractor = hydra.utils.instantiate(cfg_resnet)

        classifier = nn.Linear(1000, num_classes)

        self.model = nn.Sequential(feature_extractor, classifier)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
