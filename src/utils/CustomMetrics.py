import torch


class MultiLabelAccuracy:
    """
    This is wrapped in a class, so it can be called the same way as torch internal accuracies.
    """
    def __call__(self, predictions, targets, *args, **kwargs):
        prediction_is_in_target = torch.any(torch.eq(targets, predictions), dim=1)
        return sum(prediction_is_in_target) / len(prediction_is_in_target)