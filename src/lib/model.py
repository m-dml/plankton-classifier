from dataclasses import dataclass


@dataclass
class FFWModel:
    _target_: str = "src.models.ffw_model.FFW_Net"
    width: int = 20


@dataclass
class UnconditionalFFWModel:
    _target_: str = "src.models.unconditional_generator.UnconditionalFFWNet"
    width: int = 20
