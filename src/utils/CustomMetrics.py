import torch


class MultiLabelAccuracy:
    """This is wrapped in a class, so it can be called the same way as torch
    internal accuracies."""

    def __init__(self, weighted=True):
        self.weighted = weighted

    def __call__(self, predictions, targets, n_labels=None, *args, **kwargs):
        if predictions.requires_grad:
            raise AttributeError(
                "Predictions are not allowed to have required_grad=True when calculating multi-label accuracy."
            )
        if targets.requires_grad:
            raise AttributeError(
                "Targets are not allowed to have required_grad=True when calculating multi-label accuracy."
            )
        if self.weighted and (n_labels is None):
            raise ValueError("When using weighted Acc.: n_labels can not be None.")

        if self.weighted:
            return self._weighted_acc(predictions, targets, n_labels)
        else:
            return self._unweighted_acc(predictions, targets)

    @staticmethod
    def _unweighted_acc(predictions, targets):
        prediction_is_in_target = torch.any(torch.eq(targets, predictions), dim=1)
        return sum(prediction_is_in_target) / len(prediction_is_in_target)

    @staticmethod
    def _weighted_acc(predictions, targets, n_labels):
        def apply_along_axis(function, x, axis: int = 0, **kwargs):
            return torch.stack([function(x_i, **kwargs) for x_i in torch.unbind(x, dim=axis)], dim=axis)

        bins = apply_along_axis(
            torch.histc, targets.float(), axis=0, bins=n_labels, min=0, max=n_labels - 1
        ) / targets.size(1)
        weights = torch.max(bins, dim=1)
        result_tensor = bins[torch.arange(0, len(bins)), predictions.squeeze()].float() / weights.values
        return result_tensor.mean()
