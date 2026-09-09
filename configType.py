from ShallowNet import shallowCNN
from ENet import ENet

from dataclasses import dataclass
from typing import Tuple, Optional, List

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
    loss_fn: str # Options: 'ce', 'weighted_ce', 'ce_dice', 'weighted_ce_dice', 'focal', 'weighted_focal', 'focal_dice'
    focal_gamma: float = 1.0  # For 'focal'
    class_weights: Optional[List[float]] = None  # For 'weighted_ce' and 'weighted_focal'

NETWORKS = {
    'shallowCNN': shallowCNN,
    'ENet': ENet
}
