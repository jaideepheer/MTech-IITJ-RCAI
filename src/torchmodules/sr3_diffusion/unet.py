# See: https://arxiv.org/abs/2104.07636v2

from typing import List

import math
import torch.nn as nn
import torch

from src.torchmodules.lazy_modules import LazyLazyConv2d


class PositionalEncoding(nn.Module):
    """
    Transformer positional encoding.

    Init. Parameters
    ---
    encoding_dim: torch.Tensor
        The dimension of each positional encoding.
    linear_scale: int = 5000
        The scaling constant to multiply exponents with before applying sin/cos.

    Forward Parameters
    ---
    x: torch.Tensor
        A tensor of shape (b, 1) containing noise levels for each element.

    Returns
    ---
    torch.Tensor: A tensor of shape (b, self.encoding_dim) representing positionally encoded data.

    See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py
    See: https://github.com/ivanvovk/WaveGrad/blob/721c37c216132a2ef0a16adc38439f993998e0b7/model/linear_modulation.py#L12
    See: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    """

    def __init__(self, encoding_dim: int, linear_scale: int = 5000):
        super().__init__()
        if encoding_dim % 2 != 0:
            raise Exception(
                "Positional encoding requires the number of encoding dim. to be divisible by 2."
            )
        self.encoding_dim_half = encoding_dim // 2
        self.linear_scale = linear_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        exponents = (
            torch.arange(
                end=self.encoding_dim_half,
                dtype=x.dtype,
                device=x.device,
            )
            / self.encoding_dim_half
        )
        # exponents shape: (b,)
        exponents = torch.exp(-math.log(1e4) * exponents)
        exponents = self.linear_scale * (x * exponents)
        # unlike https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ we do not intertwine sin and cos values
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class FeatureWiseLinearModulation(nn.modules.lazy.LazyModuleMixin, nn.Module):
    """
    Feature Wise Linear Modulation (FiLM) module, using a single nn.Linear layer.

    See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/skip_unet.py
    See: https://distill.pub/2018/feature-wise-transformations/

    Init. Parameters
    ----------
    affine: bool = False
        If True, the transformation is affine.

    Forward Parameters
    ---
    x: torch.Tensor
        The data having shape (b, c, h, w).
    conditioning: torch.Tensor
        The conditioning tensor having shape (b, dim)

    Returns
    ---
    torch.Tensor: The feature wise transformed data having shape (b, c, h, w).
    """

    def __init__(self, affine: bool = False) -> None:
        super().__init__()
        self.affine = affine
        # applied on conditioning tensor of shape (b, dim)
        self.condition_transformer = nn.LazyLinear(0, bias=False)

    def initialize_parameters(self, x: torch.Tensor, conditioning: torch.Tensor):
        self.condition_transformer.out_features = (
            (x.shape[-3] * 2) if self.affine is True else x.shape[-3]
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        conditioning = self.condition_transformer(conditioning)  # shape: (b, c or 2c)
        conditioning = conditioning.view((x.shape[0], -1, 1, 1))
        if self.affine is True:
            scale, bias = conditioning.chunk(2, dim=-1)
            x = (1.0 + scale) * x + bias
        else:
            x = x + conditioning
        return x


class BigGANResnetBlockUp(nn.Module):
    """
    Resnet blocks used in BigGAN with support for conditional tensor.

    See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/skip_unet.py
    See: https://github.com/sxhxliang/BigGAN-pytorch

    Init. Parameters
    ----------
    out_channels: int
        If `0` out channels == input channels.
    dropout: float = 0.0
    apply_scaling: bool = True
    affine_conditional_modulation: bool = False

    Forward Parameters
    ---
    x: torch.Tensor
        The data having shape (b, c, h, w).
    condition: torch.Tensor
        The conditioning tensor having shape (b, dim)

    Returns
    ---
    torch.Tensor: The data tensor after applying this module.
    """

    def __init__(
        self,
        out_channels: int = None,
        dropout: float = 0.0,
        apply_scaling: bool = True,
        affine_conditional_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            (nn.Upsample(scale_factor=2) if apply_scaling is True else nn.Identity()),
            LazyLazyConv2d(out_channels, 3, 1, 1),
        )
        self.conditional_modulator = FeatureWiseLinearModulation(
            affine_conditional_modulation
        )
        self.post = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            (nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()),
            LazyLazyConv2d(out_channels, 3, 1, 1),
        )
        self.skip = nn.Sequential(
            (nn.Upsample(scale_factor=2) if apply_scaling is True else nn.Identity()),
            LazyLazyConv2d(out_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.pre(x)
        x = self.conditional_modulator(x, condition)
        return self.post(x) + self.skip(x_in)


class BigGANResnetBlockDown(nn.Module):
    """
    Resnet blocks used in BigGAN with support for conditional tensor.

    See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/skip_unet.py
    See: https://github.com/sxhxliang/BigGAN-pytorch

    Init. Parameters
    ----------
    out_channels: int
        If `0` out channels == input channels.
    dropout: float = 0.0
    apply_scaling: bool = True
    affine_conditional_modulation: bool = False

    Forward Parameters
    ---
    x: torch.Tensor
        The data having shape (b, c, h, w).
    condition: torch.Tensor
        The conditioning tensor having shape (b, dim)

    Returns
    ---
    torch.Tensor: The data tensor after applying this module.
    """

    def __init__(
        self,
        out_channels: int = None,
        dropout: float = 0.0,
        apply_scaling: bool = True,
        affine_conditional_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.ReLU(),
            LazyLazyConv2d(out_channels, 3, 1, 1),
        )
        self.conditional_modulator = FeatureWiseLinearModulation(
            affine_conditional_modulation
        )
        self.post = nn.Sequential(
            nn.ReLU(),
            (nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()),
            LazyLazyConv2d(out_channels, 3, 1, 1),
            (nn.AvgPool2d(2, 2, 0) if apply_scaling is True else nn.Identity()),
        )
        self.skip = nn.Sequential(
            LazyLazyConv2d(out_channels, 1, 1),
            (nn.AvgPool2d(2, 2, 0) if apply_scaling is True else nn.Identity()),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.pre(x)
        x = self.conditional_modulator(x, condition)
        return self.post(x) + self.skip(x_in)


class UNetWithNoiseLevel(nn.Module):
    """
    The UNet model used in SR3.
    See: Fig. A.1 in https://arxiv.org/abs/2104.07636v2

    Code: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/skip_unet.py

    Init. Parameters
    ---

    """

    def __init__(
        self,
        out_channels: int,
        inner_channels: int = 32,
        channel_multipliers: List[int] = [1, 2, 4, 8, 8],
        resnet_blocks_per_unet_layer: int = 3,
        dropout: float = 0.0,
        skip_connect_scaling: float = 1.0 / math.sqrt(2),
        affine_conditional_modulation: bool = False,
    ) -> None:
        super().__init__()
        self.skip_connect_scaling = skip_connect_scaling
        opts = {
            "dropout": dropout,
            "affine_conditional_modulation": affine_conditional_modulation,
        }
        # make downscaling network
        downs = []
        for idx, level in enumerate(channel_multipliers):
            # make this level's blocks
            blocks = [
                BigGANResnetBlockDown(
                    out_channels=(inner_channels * level),
                    apply_scaling=False,
                    **opts,
                )
                for _ in range(resnet_blocks_per_unet_layer - 1)
            ]
            if idx != len(channel_multipliers) - 1:
                # only add scaling block if not on last layer
                blocks.append(
                    BigGANResnetBlockDown(
                        out_channels=(inner_channels * channel_multipliers[idx + 1]),
                        apply_scaling=True,
                        **opts,
                    )
                )
            # add layer blocks
            downs.extend(blocks)

        # make upscaling network
        ups = []
        multi = list(reversed(channel_multipliers))
        for idx, level in enumerate(multi):
            # make this level's blocks
            blocks = [
                BigGANResnetBlockUp(
                    out_channels=(inner_channels * level),
                    apply_scaling=False,
                    **opts,
                )
                for _ in range(resnet_blocks_per_unet_layer - 1)
            ]
            if idx != len(multi) - 1:
                # only add scaling block if not on first layer
                blocks.append(
                    BigGANResnetBlockUp(
                        out_channels=(inner_channels * multi[idx + 1]),
                        apply_scaling=True,
                        **opts,
                    )
                )
            # add layer blocks
            ups.extend(blocks)

        # make module lists
        self.ups = nn.ModuleList(ups)
        self.downs = nn.ModuleList(downs)

        # make other modules
        self.condition_encoder = PositionalEncoding(inner_channels * 4)
        self.pre = LazyLazyConv2d(inner_channels, kernel_size=3, padding=1)
        self.post = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            LazyLazyConv2d(out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = self.condition_encoder(condition)
        x = self.pre(x)
        # start downscale
        feat = []
        for m in self.downs:
            x = m(x, condition)
            feat.append(x)
        # start upscale
        for m in self.ups:
            skip = feat.pop() * self.skip_connect_scaling
            x = m(x + skip, condition)
        # final conv
        return self.post(x)
