import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from ENet import ENet
from UNet import UNet
from configType import TrainConfig
from infer import build_net
from losses import CrossEntropy


def config_for(net_name: str, *, in_slices: int = 1, kernels: int = 8, factor: int = 2) -> TrainConfig:
    return TrainConfig(dataset="SEGTHOR", mode="full", K=5, net_name=net_name,
                       B=8, kernels=kernels, factor=factor, lr=0.0005,
                       betas=(0.9, 0.999), epochs=25, num_workers=0,
                       temperature=1.0, optimizer="Adam", seed=123, patience=-1,
                       in_slices=in_slices)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Keep this CPU smoke suite fast and avoid oversubscribing shared nodes.
        torch.set_num_threads(1)

    def test_baseline_and_large_enet_forward(self) -> None:
        inputs = torch.randn(1, 1, 16, 16)
        baseline = ENet(1, 5, kernels=8, factor=2).eval()
        large = ENet(1, 5, kernels=16, factor=4).eval()

        with torch.no_grad():
            self.assertEqual(tuple(baseline(inputs).shape), (1, 5, 16, 16))
            self.assertEqual(tuple(large(inputs).shape), (1, 5, 16, 16))

        self.assertGreater(sum(p.numel() for p in large.parameters()),
                           sum(p.numel() for p in baseline.parameters()))

    def test_unet_one_and_three_channel_forward_and_backward(self) -> None:
        loss_fn = CrossEntropy(idk=list(range(5)))
        for in_slices in [1, 3]:
            with self.subTest(in_slices=in_slices):
                net = UNet(in_slices, 5, kernels=32, factor=4)
                net.init_weights()
                images = torch.randn(2, in_slices, 16, 16)
                labels = torch.randint(0, 5, (2, 16, 16))
                targets = F.one_hot(labels, num_classes=5).permute(0, 3, 1, 2).float()

                logits = net(images)
                self.assertEqual(tuple(logits.shape), (2, 5, 16, 16))
                loss = loss_fn(F.softmax(logits, dim=1), targets)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(p.grad is not None for p in net.parameters()))

    def test_unet_checkpoint_rebuilds_through_inference_path(self) -> None:
        config = config_for("UNet", kernels=32, factor=4)
        net = UNet(1, 5, kernels=config.kernels, factor=config.factor)
        net.init_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            weights = Path(tmpdir) / "weights.pt"
            torch.save(net.state_dict(), weights)
            restored = build_net(config, weights, torch.device("cpu"))

        self.assertIsInstance(restored, UNet)
        self.assertFalse(restored.training)
        with torch.no_grad():
            self.assertEqual(tuple(restored(torch.randn(1, 1, 16, 16)).shape), (1, 5, 16, 16))

    def test_parameter_count_ordering(self) -> None:
        baseline = ENet(1, 5, kernels=8, factor=2)
        large = ENet(1, 5, kernels=16, factor=4)
        unet = UNet(1, 5, kernels=32, factor=4)
        counts = [sum(p.numel() for p in model.parameters()) for model in [baseline, large, unet]]
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])


if __name__ == "__main__":
    unittest.main()
