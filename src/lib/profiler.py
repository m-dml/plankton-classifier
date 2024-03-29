from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PytorchProfiler:
    _target_: str = "pytorch_lightning.profiler.PyTorchProfiler"
    filename: str = "profile.txt"
    profiler_kwargs: Dict = field(default_factory=lambda: dict(profile_memory=True))


@dataclass
class NoProfiler:
    _target_: str = "pytorch_lightning.profiler.PassThroughProfiler"
