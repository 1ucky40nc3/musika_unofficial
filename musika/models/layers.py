from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import torch
from torch import nn

from torchvision.ops import SqueezeExcitation


class AddNoise(nn.Module):
    def __init__(self):
        self.bias = nn.Parameter(
            torch.rand(1),
            requires_grad=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.normal()


def conv_kwargs(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int],
    stride: Tuple[int],
    padding: Union[str, Tuple[int]],
    **kwargs
) -> Dict[str, Any]:
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding
    } 

def conv_util(
    in_channels,
    out_channels,
    kernel_size=(1, 3),
    stride=(1, 1),
    noise=False,
    upsample=False,
    padding="same",
    bn=True
) -> nn.Sequential:
    layers = []

    if upsample:
        layers.append(
            nn.ConvTranspose2d(**conv_kwargs(**locals()))
        )
    else:
        layers.append(
            nn.Conv2d(**conv_kwargs(**locals()))
        )

    if noise:
        pass