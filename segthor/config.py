from segthor.models.ShallowNet import shallowCNN
from segthor.models.ENet import ENet
from segthor.models.UNet import UNet

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
    # Passed explicitly so AdamW never falls back to torch's implicit 0.01;
    # 0.0 keeps every earlier (Adam) run unchanged.
    weight_decay: float = 0.0
    # None keeps the constant learning rate of every earlier run; "cosine"
    # anneals lr -> lr_min over `epochs` (one step per epoch).
    lr_scheduler: Optional[str] = None
    lr_min: float = 0.0
    # 1 preserves the original 2D baseline; 3 uses [z-1, z, z+1] as channels.
    in_slices: int = 1
    # Augmentation (1.5 ablation); defaults reproduce the E001 baseline (off).
    augment: bool = False
    aug_rotation: float = 15.0
    aug_scale_min: float = 0.9
    aug_scale_max: float = 1.1
    aug_intensity: float = 0.1
    aug_elastic_alpha: float = 10.0
    aug_elastic_sigma: float = 4.0
    # Optional scalar-only Weights & Biases logging. Disabled by default so
    # existing experiments do not require a W&B account or the wandb package.
    wandb_enabled: bool = False
    wandb_project: str = "AI4M"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        assert self.in_slices in [1, 3], \
            f"Only 1 and 3 input slices are supported, got {self.in_slices}"
        assert self.lr_scheduler in [None, "cosine"], \
            f"Unsupported lr_scheduler '{self.lr_scheduler}'"
        assert self.weight_decay >= 0, self.weight_decay

NETWORKS = {
    'shallowCNN': shallowCNN,
    'ENet': ENet,
    'UNet': UNet,
}
