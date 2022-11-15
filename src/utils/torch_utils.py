from torch import nn


def concat_feature_extractor_and_classifier(feature_extractor, classifier) -> nn.Sequential:
    """Concatenates a feature extractor and a classifier into a single model.

    Args:
        feature_extractor (torch.nn.Module): A pytorch model that extracts features from an input.
        classifier (torch.nn.Module): A pytorch model that takes the features extracted by the feature_extractor for
            classification.

    """
    model = nn.Sequential()
    model.add_module("feature_extractor", feature_extractor)
    model.add_module("classifier", classifier)
    return model
