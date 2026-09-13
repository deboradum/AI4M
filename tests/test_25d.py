import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ENet import ENet
from dataset import SliceDataset
from losses import CrossEntropy


def image_transform(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32)[None] / 255)


def gt_transform(image: Image.Image) -> torch.Tensor:
    height, width = np.asarray(image).shape
    gt = torch.zeros((2, height, width), dtype=torch.int32)
    gt[0] = 1
    return gt


class Test25D(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for kind in ["img", "gt"]:
            (self.root / "train" / kind).mkdir(parents=True)

        # Patient 01 has three slices. Patient 02 is deliberately adjacent in
        # lexical order to catch accidental cross-patient neighbours.
        for stem, value in [("Patient_01_0000", 10), ("Patient_01_0001", 20),
                            ("Patient_01_0002", 30), ("Patient_02_0000", 200)]:
            image = Image.fromarray(np.full((32, 32), value, dtype=np.uint8))
            image.save(self.root / "train" / "img" / f"{stem}.png")
            Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(
                self.root / "train" / "gt" / f"{stem}.png")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def dataset(self, in_slices: int) -> SliceDataset:
        return SliceDataset("train", self.root, image_transform, gt_transform, in_slices=in_slices)

    def samples(self, dataset: SliceDataset) -> dict[str, dict]:
        return {dataset[i]["stems"]: dataset[i] for i in range(len(dataset))}

    def test_single_slice_shape(self) -> None:
        sample = self.samples(self.dataset(1))["Patient_01_0001"]
        self.assertEqual(sample["images"].shape, (1, 32, 32))
        self.assertEqual(sample["gts"].shape, (2, 32, 32))
        self.assertAlmostEqual(sample["images"][0, 0, 0].item(), 20 / 255)

    def test_three_slice_shape_and_boundaries(self) -> None:
        samples = self.samples(self.dataset(3))
        self.assertEqual(samples["Patient_01_0001"]["images"].shape, (3, 32, 32))
        self.assertEqual(samples["Patient_01_0001"]["gts"].shape, (2, 32, 32))

        for stem, expected in [("Patient_01_0000", [10, 10, 20]),
                               ("Patient_01_0002", [20, 30, 30])]:
            values = [round(255 * channel[0, 0].item()) for channel in samples[stem]["images"]]
            self.assertEqual(values, expected)

    def test_neighbours_do_not_cross_patients(self) -> None:
        last_patient_one = self.samples(self.dataset(3))["Patient_01_0002"]["images"]
        values = [round(255 * channel[0, 0].item()) for channel in last_patient_one]
        self.assertEqual(values, [20, 30, 30])
        self.assertNotIn(200, values)

    def test_enet_forward_one_and_three_channels(self) -> None:
        for channels in [1, 3]:
            net = ENet(channels, 5, kernels=4, factor=2).eval()
            with torch.no_grad():
                output = net(torch.rand(1, channels, 32, 32))
            self.assertEqual(output.shape, (1, 5, 32, 32))

    def test_three_channel_training_step(self) -> None:
        net = ENet(3, 5, kernels=4, factor=2)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        image = torch.rand(1, 3, 32, 32)
        target_classes = torch.randint(5, (1, 32, 32))
        gt = F.one_hot(target_classes, num_classes=5).permute(0, 3, 1, 2).to(torch.int32)

        optimizer.zero_grad()
        probabilities = F.softmax(net(image), dim=1)
        loss = CrossEntropy(idk=list(range(5)))(probabilities, gt)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
