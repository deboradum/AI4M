#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class WeightedCrossEntropy():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        # Expects a list of weights matching the length of idk (e.g., [0.1, 10.0, 10.0, 10.0])
        self.weights = kwargs.get('class_weights', None)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        if self.weights is not None:
            # Broadcast weights to match batch and spatial dimensions
            w = torch.tensor(self.weights, device=mask.device, dtype=mask.dtype)
            w = w.view(1, len(self.idk), 1, 1)
            mask = mask * w

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10
        return loss


class CEDiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.ce = CrossEntropy(idk=self.idk)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        # Cross Entropy loss
        ce_loss = self.ce(pred_softmax, weak_target)

        # Soft Dice Loss
        p = pred_softmax[:, self.idk, ...]
        g = weak_target[:, self.idk, ...].float()

        eps = 1e-5
        intersection = einsum("bkwh,bkwh->bk", p, g)
        p_sum = einsum("bkwh->bk", p)
        g_sum = einsum("bkwh->bk", g)

        dice = (2. * intersection + eps) / (p_sum + g_sum + eps)
        dice_loss = 1. - dice.mean()

        return ce_loss + dice_loss


class WeightedCEDiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.weighted_ce = WeightedCrossEntropy(idk=self.idk, class_weights=kwargs.get('class_weights', None))
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        ce_loss = self.weighted_ce(pred_softmax, weak_target)

        p = pred_softmax[:, self.idk, ...]
        g = weak_target[:, self.idk, ...].float()

        eps = 1e-5
        intersection = einsum("bkwh,bkwh->bk", p, g)
        p_sum = einsum("bkwh->bk", p)
        g_sum = einsum("bkwh->bk", g)

        dice = (2. * intersection + eps) / (p_sum + g_sum + eps)
        dice_loss = 1. - dice.mean()

        return ce_loss + dice_loss

class FocalLoss():
    def __init__(self, **kwargs):
        # Extract idk and focal hyperparameters from kwargs
        self.idk = kwargs['idk']
        self.gamma = kwargs.get('focal_gamma', 2.0)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        # Extract predictions and targets for the relevant classes
        p = torch.clamp(pred_softmax[:, self.idk, ...], min=1e-7, max=1. - 1e-7)
        log_p = p.log()

        mask = weak_target[:, self.idk, ...].float()

        # Calculate the focal modulating factor: (1 - p)^gamma
        focal_weight = (1 - p) ** self.gamma

        # Apply the focal weight to the log probabilities
        weighted_log_p = focal_weight * log_p

        # Calculate the masked loss using einsum
        loss = - einsum("bkwh,bkwh->", mask, weighted_log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialFocalLoss(FocalLoss):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class WeightedFocalLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.gamma = kwargs.get('focal_gamma', 2.0)
        self.alpha = kwargs.get('class_weights', None)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        p = torch.clamp(pred_softmax[:, self.idk, ...], min=1e-7, max=1. - 1e-7)
        log_p = p.log()

        mask = weak_target[:, self.idk, ...].float()

        # Apply class weights (alpha)
        if self.alpha is not None:
            w = torch.tensor(self.alpha, device=mask.device, dtype=mask.dtype)
            w = w.view(1, len(self.idk), 1, 1)
            mask = mask * w

        # Apply focal weight (gamma)
        focal_weight = (1 - p) ** self.gamma
        weighted_log_p = focal_weight * log_p

        loss = - einsum("bkwh,bkwh->", mask, weighted_log_p)
        loss /= mask.sum() + 1e-10

        return loss


class FocalDiceLoss():
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        self.focal = FocalLoss(idk=self.idk, focal_gamma=kwargs.get('focal_gamma', 2.0))
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        # Focal Loss
        f_loss = self.focal(pred_softmax, weak_target)

        # Soft Dice Loss
        p = pred_softmax[:, self.idk, ...]
        g = weak_target[:, self.idk, ...].float()

        eps = 1e-5
        intersection = einsum("bkwh,bkwh->bk", p, g)
        p_sum = einsum("bkwh->bk", p)
        g_sum = einsum("bkwh->bk", g)

        dice = (2. * intersection + eps) / (p_sum + g_sum + eps)
        dice_loss = 1. - dice.mean()

        return f_loss + dice_loss
