#!/usr/bin/env python3

"""Plain configurable 3D U-Net for volumetric semantic segmentation.

Same interface and layout as the 2D ``UNet``: ``Model(in_dim, out_dim,
kernels=..., factor=...)`` where ``kernels`` is the base channel width and
``factor`` the number of encoder levels.  Differences from the 2D network:
3x3x3 convolutions, 2x2x2 pooling/upsampling, and InstanceNorm instead of
BatchNorm, since 3D batches hold only a couple of patches.
"""

import torch
import torch.nn as nn
from torch import Tensor


def random_weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight.data, a=0.25)  # PReLU's initial slope
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.InstanceNorm3d) and module.affine:
        module.weight.data.fill_(1)
        module.bias.data.zero_()


class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolution, InstanceNorm, PReLU operations."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class UNet3D(nn.Module):
    """A plain 3D U-Net with max-pooling, skip connections, and transposed convolutions."""
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        base_channels: int = kwargs.get("kernels", 32)
        levels: int = kwargs.get("factor", 4)
        max_channels: int = kwargs.get("max_channels", 320)
        assert base_channels > 0, f"U-Net base channels must be positive, got {base_channels}"
        assert levels >= 1, f"U-Net levels must be at least one, got {levels}"

        # Capped like nnU-Net: 3D feature maps get expensive fast at depth
        channels = [min(base_channels * 2 ** level, max_channels) for level in range(levels)]
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder = nn.ModuleList()
        previous_channels = in_dim
        for out_channels in channels:
            self.encoder.append(ConvBlock3D(previous_channels, out_channels))
            previous_channels = out_channels

        bottleneck_channels = min(channels[-1] * 2, max_channels)
        self.bottleneck = ConvBlock3D(channels[-1], bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        previous_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            self.upconvs.append(nn.ConvTranspose3d(previous_channels, skip_channels,
                                                   kernel_size=2, stride=2, bias=False))
            self.decoder.append(ConvBlock3D(skip_channels * 2, skip_channels))
            previous_channels = skip_channels

        self.final = nn.Conv3d(channels[0], out_dim, kernel_size=1)
        self.base_channels = base_channels
        self.levels = levels
        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with "
              f"channels={channels}, bottleneck={bottleneck_channels}")

    def forward(self, x: Tensor) -> Tensor:
        divisor = 2 ** self.levels
        assert all(s % divisor == 0 for s in x.shape[-3:]), \
            f"3D U-Net input spatial shape {tuple(x.shape[-3:])} must be divisible by {divisor}"

        skips: list[Tensor] = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        for upconv, block, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            x = upconv(x)
            assert x.shape[-3:] == skip.shape[-3:], (x.shape, skip.shape)
            x = block(torch.cat((x, skip), dim=1))

        return self.final(x)

    def init_weights(self, *args, **kwargs) -> None:
        self.apply(random_weights_init)
