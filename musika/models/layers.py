from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import torch
from torch import nn

from torchvision.ops import SqueezeExcitation

import einops




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


class Swish(nn.Module):
    def foward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


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
        layers.append(nn.ConvTranspose2d(**conv_kwargs(**locals())))
    else:
        layers.append(nn.Conv2d(**conv_kwargs(**locals())))

    if noise:
        layers.append(nn.Dropout(p=0.1))
    
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    
    layers.append(Swish())

    return nn.Sequential(*layers)


class Encoder2(nn.Module):
    def __init__(
        self,
        shape: int,
        dim: int = 128,
    ) -> None:
        kwargs = dict(stride=(1, 1), padding="valid")
        self.g0 = conv_util(dim      , 256      , kernel_size=(1, 1), **kwargs)
        self.g1 = conv_util(256      , 256 + 256, kernel_size=(1, 3), **kwargs)
        self.g2 = conv_util(256 + 256, 512 + 128, kernel_size=(1, 3), **kwargs)
        self.g3 = conv_util(512 + 128, 512 + 128, kernel_size=(1, 1), **kwargs)
        self.g4 = conv_util(512 + 128, 512 + 128, kernel_size=(1, 3), **kwargs)
        self.g5 = conv_util(512 + 256, 512 + 256, kernel_size=(1, 2), **kwargs)

        self.g = nn.Conv2d(
            in_channels=512 + 256, 
            out_channels=64, 
            kernel_size=(1, 1), 
            stride=1,
            name="cbottle")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.split(x, x.shape[-2] // 16, -2)
        x = torch.concat(x, 1)

        x = self.g0(x)
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.g4(x)
        x = self.g5(x)
        
        x = self.g(x)
        x = torch.tanh(x)
        
        x = torch.split(x, x.shape[1] // 16, 1)
        x = torch.concat(x, -2)
        x = torch.split(x, x.shape[-2] // 2, -2)
        x = torch.concat(x, 1)

        return x.float()


class Decoder2(nn.Module):
    def __init__(
        self,
        shape,
        dim,
        bottledim
    ) -> None:
        kwargs = dict(kernel_size=(1, 4), noise=True)
        self.g4 = conv_util(bottledim      , 512 + 128 + 128, stride=(1, 1), upsample=False, **kwargs)
        self.g3 = conv_util(512 + 128 + 128, 512 + 128 + 128, stride=(1, 2), upsample=True , **kwargs)
        self.g2 = conv_util(512 + 128 + 128, 512 + 128      , stride=(1, 2), upsample=True , **kwargs)
        self.g1 = conv_util(512 + 128      , 512            , stride=(1, 1), upsample=False, **kwargs)
        self.g0 = conv_util(512            , 256 + 128      , stride=(1, 2), upsample=True , **kwargs)

        self.g = nn.Conv2d(
            in_channels=256 + 128, 
            out_channels=dim,
            kernel_size=(1, 1),
            strides=1,
            padding="same")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.g4(x)
        x = self.g3(x)
        x = self.g2(x)
        x = self.g1(x)
        x = self.g0(x)

        x = self.g(x)
        x = torch.tanh(x)

        x = torch.split(x, x.shape[1] // 2, 1)
        x = torch.concat(x, -2)

        return x.float()


class Encoder(nn.Module):
    def __init__(
        self,
        hop,
        shape
    ) -> None:
        dim = ((4 * hop) // 2) + 1

        kwargs = dict(kernel_size=(1, 1), stride=(1, 1), padding="valid")
        self.g0 = conv_util(dim               , hop * 2 + 32       , **kwargs)
        self.g1 = conv_util(hop * 2 + 32      , hop * 2 + 64       , **kwargs)
        self.g2 = conv_util(hop * 2 + 64      , hop * 2 + 64 + 64  , **kwargs)
        self.g3 = conv_util(hop * 2 + 64 + 64 , hop * 2 + 128 + 64 , **kwargs)
        self.g4 = conv_util(hop * 2 + 128 + 64, hop * 2 + 128 + 128, **kwargs)

        self.g = nn.Conv2d(
            in_channels=hop * 2 + 128 + 128,
            out_channels=128,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b c h w -> b w h c")
        
        x = self.g0(x)
        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.g4(x)

        x = self.g(x)
        x = torch.tanh(x)

        x = torch.split(x, x.shape[-2] // 2, -2)
        x = torch.concat(x, 1)

        return x.float()


class Decoder(nn.Module):
    def __init__(
        self,
        hop,
        shape,
        window
    ) -> None:
        self.shape = shape
        self.window = window
        dim = ((4 * hop) // 2) + 1
        
        kwargs = dict(noise=True)
        self.g0 = conv_util(128           , hop * 3       , kernel_size=(1, 1), stride=(1, 1), **kwargs)
        self.g1 = conv_util(hop * 3       , hop * 2       , kernel_size=(1, 3), stride=(1, 2), **kwargs)
        self.g2 = conv_util(hop * 2       , hop + hop // 2, kernel_size=(1, 3), stride=(1, 2), **kwargs)
        self.g3 = conv_util(hop + hop // 2, hop + hop // 4, kernel_size=(1, 3), strides=(1, 2), **kwargs)
        self.g4 = conv_util(hop + hop // 4, hop + hop // 2, kernel_size=(1, 4), stride=(1, 2), upsample=True, **kwargs)
        self.g5 = conv_util(hop + hop // 2, hop * 2       , kernel_size=(1, 4), stride=(1, 2), upsample=True, **kwargs)
        self.g6 = conv_util(hop * 2       , hop * 3       , kernel_size=(1, 4), stride=(1, 2), upsample=True, **kwargs)
        self.g7 = conv_util(hop * 3       , hop * 5       , kernel_size=(1, 1), stride=(1, 1), **kwargs)

        self.g = nn.Conv2d(
            in_channels=hop * 5,
            out_channels=dim * 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_g0 = self.g0(x)
        x_g1 = self.g1(x_g0)
        x_g2 = self.g2(x_g1)
        x = self.g3(x_g2)
        x = self.g4(x)
        x = self.g5(x + x_g2)
        x = self.g6(x + x_g1)
        x = self.g7(x + x_g0)

        x = self.g(x)
        x = torch.clip(x, -1., 1.)

        gf, pf = torch.split(x, x.shape[-1] // 2, -1)
        gf = torch.split(gf, gf.shape[1] // self.shape // self.window, 1)
        gf = torch.concat(gf, -2)

        pf = torch.split(pf, pf.shape[1] // self.shape // self.window, 1)
        pf = torch.concat(pf, -2)

        s = einops.rearrange(gf, "b c h w -> b h w c")
        p = einops.rearrange(pf, "b c h w -> b h w c")
        s = torch.squeeze(s).float()
        p = torch.squeeze(p).float()

        return s, p