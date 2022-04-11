from dataclasses import dataclass


@dataclass
class DDPPlugin:
    _target_: str = "pytorch_lightning.strategies.ddp.DDPStrategy"
    find_unused_parameters: bool = False


@dataclass
class SingleDevicePlugin:
    _target_: str = "pytorch_lightning.plugins.training_type.SingleDevicePlugin"
