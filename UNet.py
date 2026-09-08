#!/usr/bin/env python3

"""Plain configurable 2D U-Net for semantic segmentation.

The constructor follows the repository-wide model interface:
``Model(in_dim, out_dim, kernels=..., factor=...)``.  For U-Net, ``kernels``
is the base channel width and ``factor`` is the number of encoder levels.
This keeps model selection compatible with the existing configuration and
training/inference paths without changing ENet.
"""

import torch
import torch.nn as nn
from torch import Tensor


def random_weights_init(module: nn.Module) -> None:
    """Use the same Conv/BatchNorm initialization convention as ENet."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


class ConvBlock(nn.Module):
    """Two 3x3 convolution, BatchNorm, PReLU operations."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """A plain 2D U-Net with max-pooling, skip connections, and transposed convolutions."""
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        base_channels: int = kwargs.get("kernels", 32)
        levels: int = kwargs.get("factor", 4)
        assert base_channels > 0, f"U-Net base channels must be positive, got {base_channels}"
        assert levels >= 1, f"U-Net levels must be at least one, got {levels}"

        channels = [base_channels * 2 ** level for level in range(levels)]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = nn.ModuleList()
        previous_channels = in_dim
        for out_channels in channels:
            self.encoder.append(ConvBlock(previous_channels, out_channels))
            previous_channels = out_channels

        bottleneck_channels = channels[-1] * 2
        self.bottleneck = ConvBlock(channels[-1], bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        previous_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(previous_channels, skip_channels,
                                                   kernel_size=2, stride=2, bias=False))
            self.decoder.append(ConvBlock(skip_channels * 2, skip_channels))
            previous_channels = skip_channels

        self.final = nn.Conv2d(channels[0], out_dim, kernel_size=1)
        self.base_channels = base_channels
        self.levels = levels
        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with "
              f"base_channels={base_channels}, levels={levels}")

    def forward(self, x: Tensor) -> Tensor:
        spatial_divisor = 2 ** self.levels
        height, width = x.shape[-2:]
        assert height % spatial_divisor == 0 and width % spatial_divisor == 0, \
            f"U-Net input spatial shape {(height, width)} must be divisible by {spatial_divisor}"

        skips: list[Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for upconv, block, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = upconv(x)
            assert x.shape[-2:] == skip.shape[-2:], (x.shape, skip.shape)
            x = block(torch.cat((x, skip), dim=1))

        return self.final(x)

    def init_weights(self, *args, **kwargs) -> None:
        self.apply(random_weights_init)
