import unittest
from pathlib import Path

from segthor.config import TrainConfig
from segthor.wandb_logger import finish_run, init_run, log_epoch


def config_for_wandb(*, enabled: bool) -> TrainConfig:
    return TrainConfig(dataset="SEGTHOR", mode="full", K=5, net_name="ENet",
                       B=8, kernels=8, factor=2, lr=0.0005,
                       betas=(0.9, 0.999), epochs=25, num_workers=0,
                       temperature=1.0, optimizer="Adam", seed=123, patience=-1,
                       loss_fn="ce", wandb_enabled=enabled,
                       wandb_project="AI4M", wandb_run_name="unit-test",
                       wandb_tags=["test"])


class FakeRun:
    def __init__(self) -> None:
        self.metrics: list[dict] = []
        self.finished = False

    def log(self, metrics: dict) -> None:
        self.metrics.append(metrics)

    def finish(self) -> None:
        self.finished = True


class FakeWandb:
    def __init__(self) -> None:
        self.kwargs: dict | None = None
        self.run = FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


class TestWandbLogger(unittest.TestCase):
    def test_disabled_logging_does_not_initialize_wandb(self) -> None:
        fake = FakeWandb()
        run = init_run(config_for_wandb(enabled=False), Path("results/test"), wandb_module=fake)
        self.assertIsNone(run)
        self.assertIsNone(fake.kwargs)

    def test_enabled_logging_is_scalar_only_and_finishes(self) -> None:
        fake = FakeWandb()
        config = config_for_wandb(enabled=True)
        run = init_run(config, Path("results/test"), wandb_module=fake)

        self.assertIs(run, fake.run)
        self.assertEqual(fake.kwargs["project"], "AI4M")
        self.assertEqual(fake.kwargs["name"], "unit-test")
        self.assertFalse(fake.kwargs["save_code"])
        self.assertEqual(fake.kwargs["config"]["dataset"], "SEGTHOR")

        log_epoch(run, {"epoch": 1, "val/dice_foreground": 0.5})
        finish_run(run)
        self.assertEqual(fake.run.metrics, [{"epoch": 1, "val/dice_foreground": 0.5}])
        self.assertTrue(fake.run.finished)
