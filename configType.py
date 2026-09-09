from ShallowNet import shallowCNN
from ENet import ENet
from UNet import UNet

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    dataset: str
    mode: str
    K: int
    net_name: str
    B: int
    kernels: int
    factor: int
    lr: float
    betas: Tuple[float, float]
    epochs: int
    num_workers: int
    temperature: float
    optimizer: str
    seed: int
    patience: int
    # 1 preserves the original 2D baseline; 3 uses [z-1, z, z+1] as channels.
    in_slices: int = 1

    def __post_init__(self) -> None:
        assert self.in_slices in [1, 3], \
            f"Only 1 and 3 input slices are supported, got {self.in_slices}"

NETWORKS = {
    'shallowCNN': shallowCNN,
    'ENet': ENet,
    'UNet': UNet,
}
