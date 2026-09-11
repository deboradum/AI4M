import unittest

import numpy as np
import torch

from dataset import AugParams, augment_sample
from slice_segthor import norm_arr, norm_window


class TestWindow(unittest.TestCase):
    def test_clip_and_scale(self):
        img = np.array([[-1200, -1000, 0, 1000, 2000]], dtype=np.int16)
        res = norm_window(img, -1000, 1000)
        self.assertEqual(res.dtype, np.uint8)
        # Below/above the window saturates to the window edges
        self.assertEqual(res[0, 0], 0)
        self.assertEqual(res[0, 1], 0)
        self.assertEqual(res[0, 3], 255)
        self.assertEqual(res[0, 4], 255)
        # Midpoint of [-1000, 1000] maps to ~127
        self.assertTrue(abs(int(res[0, 2]) - 127) <= 1)

    def test_default_norm_unchanged(self):
        img = np.array([[10, 20, 30]], dtype=np.int16)
        res = norm_arr(img)
        self.assertEqual(res.min(), 0)
        self.assertEqual(res.max(), 255)

    def test_window_rejects_bad_range(self):
        with self.assertRaises(AssertionError):
            norm_window(np.zeros((2, 2), dtype=np.int16), 100, -100)


def _make_sample(K: int = 5, size: int = 64):
    img = torch.rand(3, size, size)
    gt = torch.zeros(K, size, size, dtype=torch.int32)
    gt[0] = 1
    gt[2, 20:30, 20:30] = 1
    gt[0, 20:30, 20:30] = 0
    return img, gt


class TestAugmentation(unittest.TestCase):
    def test_deterministic_under_seed(self):
        img, gt = _make_sample()
        torch.manual_seed(42)
        img_a, gt_a = augment_sample(img.clone(), gt.clone(), AugParams(), K=5)
        torch.manual_seed(42)
        img_b, gt_b = augment_sample(img.clone(), gt.clone(), AugParams(), K=5)
        self.assertTrue(torch.equal(img_a, img_b))
        self.assertTrue(torch.equal(gt_a, gt_b))

    def test_different_seed_differs(self):
        img, gt = _make_sample()
        torch.manual_seed(1)
        img_a, _ = augment_sample(img.clone(), gt.clone(), AugParams(), K=5)
        torch.manual_seed(2)
        img_b, _ = augment_sample(img.clone(), gt.clone(), AugParams(), K=5)
        self.assertFalse(torch.equal(img_a, img_b))

    def test_output_shapes_and_valid_labels(self):
        img, gt = _make_sample()
        torch.manual_seed(0)
        img_a, gt_a = augment_sample(img, gt, AugParams(), K=5)
        self.assertEqual(img_a.shape, img.shape)
        self.assertEqual(gt_a.shape, gt.shape)
        # Still a valid one-hot: exactly one class per pixel
        self.assertTrue(torch.equal(gt_a.sum(dim=0), torch.ones(64, 64, dtype=torch.int32)))
        self.assertTrue(img_a.min() >= 0 and img_a.max() <= 1)

    def test_all_channels_same_geometric_transform(self):
        # If channels 0..2 identical, they must stay identical after aug
        img, gt = _make_sample()
        img[1] = img[0]
        img[2] = img[0]
        torch.manual_seed(7)
        img_a, _ = augment_sample(img, gt, AugParams(intensity_shift=0.0), K=5)
        self.assertTrue(torch.allclose(img_a[0], img_a[1]))
        self.assertTrue(torch.allclose(img_a[0], img_a[2]))

    def test_zero_params_is_identity_up_to_labels(self):
        img, gt = _make_sample()
        params = AugParams(rotation_deg=0.0, scale_min=1.0, scale_max=1.0,
                           intensity_shift=0.0, elastic_alpha=0.0)
        torch.manual_seed(0)
        img_a, gt_a = augment_sample(img.clone(), gt.clone(), params, K=5)
        self.assertTrue(torch.allclose(img, img_a))
        self.assertTrue(torch.equal(gt, gt_a))


if __name__ == "__main__":
    unittest.main()
