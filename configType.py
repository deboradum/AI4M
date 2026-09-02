from ShallowNet import shallowCNN
from ENet import ENet

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

NETWORKS = {
    'shallowCNN': shallowCNN,
    'ENet': ENet
}
