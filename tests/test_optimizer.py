import math
import unittest

import torch
from torch import nn

from main import build_optimizer, build_scheduler
from segthor.config import TrainConfig


def config_for_optim(**overrides) -> TrainConfig:
    fields = dict(dataset="SEGTHOR", mode="full", K=5, net_name="UNet",
                  B=8, kernels=32, factor=4, lr=0.0005,
                  betas=(0.9, 0.999), epochs=10, num_workers=0,
                  temperature=1.0, optimizer="Adam", seed=123, patience=-1,
                  loss_fn="ce_dice")
    fields.update(overrides)
    return TrainConfig(**fields)


class TestOptimizer(unittest.TestCase):
    def test_default_is_unchanged_adam(self) -> None:
        config = config_for_optim()
        optimizer = build_optimizer(nn.Linear(2, 2), config)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertIsNone(build_scheduler(optimizer, config))

    def test_adamw_uses_configured_weight_decay(self) -> None:
        # torch's AdamW defaults to 0.01; the config value must win, including 0.
        for wd in [0.0, 1e-4]:
            optimizer = build_optimizer(nn.Linear(2, 2),
                                        config_for_optim(optimizer="AdamW", weight_decay=wd))
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            self.assertEqual(optimizer.param_groups[0]["weight_decay"], wd)

    def test_cosine_anneals_to_lr_min_over_epochs(self) -> None:
        config = config_for_optim(optimizer="AdamW", weight_decay=1e-4,
                                  lr_scheduler="cosine", epochs=10)
        optimizer = build_optimizer(nn.Linear(2, 2), config)
        scheduler = build_scheduler(optimizer, config)
        lrs = []
        for _ in range(config.epochs):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(lrs[0], config.lr)
        self.assertTrue(all(a > b for a, b in zip(lrs, lrs[1:])))
        self.assertAlmostEqual(lrs[5], config.lr * 0.5 * (1 + math.cos(math.pi * 5 / 10)))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_unknown_scheduler_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            config_for_optim(lr_scheduler="step")


if __name__ == "__main__":
    unittest.main()
