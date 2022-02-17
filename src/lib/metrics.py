from dataclasses import dataclass


@dataclass
class Accuracy:
    _target_: str = "torchmetrics.Accuracy"


@dataclass
class MultiLabelAccuracy:
    _target_: str = "src.utils.CustomMetrics.MultiLabelAccuracy"