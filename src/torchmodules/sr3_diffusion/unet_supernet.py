# See: https://arxiv.org/abs/2104.07636v2

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.torchmodules.activations import GumbelSoftmax
from src.torchmodules.lazy_modules import LazyLazyConv2d
from src.torchmodules.mixins import (
    BasicMeasurableMixin,
    BasicMeasurableWrapper,
    MeasurableModuleMixin,
)
from src.torchmodules.nas_modules import MultiOperation
from src.torchmodules.structures import MeasurableSequential, UNetLayer
from src.utils.provider import Provider, UniqueProvider

_default_conv_supernet_space = [
    {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    },
    {
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
    },
]


@UniqueProvider()
def _makeSupernetConv(
    out_channels, supernet_conv_param_space=_default_conv_supernet_space
):
    if len(supernet_conv_param_space) == 1:
        rt = LazyLazyConv2d(out_channels, **supernet_conv_param_space[0])
    else:
        rt = MultiOperation(
            [
                BasicMeasurableWrapper.wrap_module(LazyLazyConv2d(out_channels, **x))
                for x in supernet_conv_param_space
            ],
            scaling=Provider.get("scaling", None),
            scaling_activation=Provider.get(
                "scaling_activation",
                default_factory=lambda: GumbelSoftmax(tau=10.0, dim=0),
            ),
            is_scaling_learnable=True,
            execution_mode="jit",
        )
    # ensure kwargs don't cause errors
    return MeasurableSequential(rt, ignore_forward_kwargs=True)


class MeasurablePositionalEncoding(BasicMeasurableMixin, nn.Module):
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
        rt = torch.cat([exponents.sin(), exponents.cos()], dim=-1)
        return rt


class MeasurableFeatureWiseLinearModulation(
    BasicMeasurableMixin, nn.modules.lazy.LazyModuleMixin, nn.Module
):
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
    condition: torch.Tensor
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

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = self.condition_transformer(condition)  # shape: (b, c or 2c)
        condition = condition.view((x.shape[0], -1, 1, 1))
        if self.affine is True:
            scale, bias = condition.chunk(2, dim=-1)
            x = (1.0 + scale) * x + bias
        else:
            x = x + condition
        return x


class SupernetBigGANResnetBlockUp(MeasurableModuleMixin, nn.Module):
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
        Whether to apply up scaling.
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
        supernet_conv_param_space: List[Dict[str, Any]] = _default_conv_supernet_space,
    ) -> None:
        super().__init__()
        self.pre = MeasurableSequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            (nn.Upsample(scale_factor=2) if apply_scaling is True else nn.Identity()),
            _makeSupernetConv(out_channels, supernet_conv_param_space),
            ignore_forward_kwargs=True,
        )
        self.conditional_modulator = MeasurableFeatureWiseLinearModulation(
            affine_conditional_modulation
        )
        self.post = MeasurableSequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            (nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()),
            _makeSupernetConv(out_channels, supernet_conv_param_space),
            ignore_forward_kwargs=True,
        )
        self.skip = MeasurableSequential(
            (nn.Upsample(scale_factor=2) if apply_scaling is True else nn.Identity()),
            LazyLazyConv2d(out_channels, 1, 1),
            ignore_forward_kwargs=True,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.pre(x)
        x = self.conditional_modulator(x, condition)
        rt = self.post(x) + self.skip(x_in)
        return rt

    def forward_measurements(self, x: Dict[str, torch.Tensor]):
        x_in = self.skip.forward_measurements(x)
        x = self.pre.forward_measurements(x)
        x = self.conditional_modulator.forward_measurements(x)
        x = self.post.forward_measurements(x)
        return {k: (x[k] + x_in[k]) for k in x.keys()}


class SupernetBigGANResnetBlockDown(MeasurableModuleMixin, nn.Module):
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
        Whether to apply down scaling.
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
        supernet_conv_param_space: List[Dict[str, Any]] = _default_conv_supernet_space,
    ) -> None:
        super().__init__()
        self.pre = MeasurableSequential(
            nn.ReLU(),
            _makeSupernetConv(out_channels, supernet_conv_param_space),
            ignore_forward_kwargs=True,
        )
        self.conditional_modulator = MeasurableFeatureWiseLinearModulation(
            affine_conditional_modulation
        )
        self.post = MeasurableSequential(
            nn.ReLU(),
            (nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()),
            _makeSupernetConv(out_channels, supernet_conv_param_space),
            (nn.AvgPool2d(2, 2, 0) if apply_scaling is True else nn.Identity()),
            ignore_forward_kwargs=True,
        )
        self.skip = MeasurableSequential(
            LazyLazyConv2d(out_channels, 1, 1),
            (nn.AvgPool2d(2, 2, 0) if apply_scaling is True else nn.Identity()),
            ignore_forward_kwargs=True,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.pre(x)
        x = self.conditional_modulator(x, condition)
        rt = self.post(x) + self.skip(x_in)
        return rt

    def forward_measurements(self, x: Dict[str, torch.Tensor]):
        x_in = self.skip.forward_measurements(x)
        x = self.pre.forward_measurements(x)
        x = self.conditional_modulator.forward_measurements(x)
        x = self.post.forward_measurements(x)
        return {k: (x[k] + x_in[k]) for k in x.keys()}


class SupernetUNet(BasicMeasurableMixin, nn.Module):
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
        n_layers = len(channel_multipliers)

        def _make_block(layer_idx, cls, apply_scaling):
            if layer_idx < 0:
                mult = 1
            else:
                mult = channel_multipliers[layer_idx]
            return cls(
                out_channels=(inner_channels * mult),
                apply_scaling=apply_scaling,
                dropout=dropout,
                affine_conditional_modulation=affine_conditional_modulation,
            )

        def _make_layer(layer_idx=0):
            pre = MeasurableSequential(
                *[
                    _make_block(
                        layer_idx,
                        SupernetBigGANResnetBlockDown,
                        apply_scaling=False,
                    )
                    for _ in range(resnet_blocks_per_unet_layer - 1)
                ],
            )
            post = MeasurableSequential(
                *[
                    _make_block(
                        layer_idx - 1,
                        SupernetBigGANResnetBlockUp,
                        apply_scaling=False,
                    )
                    for _ in range(resnet_blocks_per_unet_layer - 1)
                ],
            )

            unet = UNetLayer(
                pre_skip=pre,
                downscale=_make_block(
                    layer_idx,
                    SupernetBigGANResnetBlockDown,
                    apply_scaling=True,
                ),
                child=(
                    # last UNet layer has a simple conv as the child
                    _make_layer(layer_idx + 1)
                    if layer_idx != n_layers - 1
                    else _makeSupernetConv(inner_channels * channel_multipliers[-1])
                ),
                upscale=_make_block(
                    # upscaling keeps channels same and lets post handle channel shrinking
                    (
                        layer_idx
                        if resnet_blocks_per_unet_layer > 1
                        else (layer_idx - 1)
                    ),
                    SupernetBigGANResnetBlockUp,
                    apply_scaling=True,
                ),
                post_skip=post,
                skip_connect_scaling=skip_connect_scaling,
            )
            return unet

        # make unet recursively
        self.unet = _make_layer()
        # make encoding module
        self.condition_encoder = MeasurablePositionalEncoding(inner_channels * 4)
        # first conv
        self.pre = BasicMeasurableWrapper.wrap_module(
            LazyLazyConv2d(inner_channels, kernel_size=3, padding=1)
        )
        # final layer for output
        self.final = MeasurableSequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            LazyLazyConv2d(out_channels, kernel_size=3, padding=1),
            ignore_forward_kwargs=True,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        # first layer and encoding
        condition = self.condition_encoder(condition)
        x = self.pre(x)
        # unet
        x = self.unet(x, condition=condition)
        # final layer
        rt = self.final(x)
        return rt

    def forward_measurements(
        self, x: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # first layer
        x = self.pre.forward_measurements(x)
        # unet
        x = self.unet.forward_measurements(x)
        # final layer
        return self.final.forward_measurements(x)


# TODO AREPL Remove

# from rich import print
# from torchinfo import summary

# # print(_makeSupernetConv(12).__class__)
# m = SupernetUNet(
#     3,
#     channel_multipliers=[1, 2],
#     resnet_blocks_per_unet_layer=2,
# )

# # print(m)

# ip = torch.ones(
#     (1, 6, 32, 32),
# )

# enc = torch.ones((1, 1))
# print("op:", m(ip, enc).shape)

# summary(m)

# print(
#     m.get_measurements(
#         {
#             "params",
#             "flops",
#             "macs",
#             "latency",
#         }
#     )
# )

# # print(
# #     m.forward_measurements(
# #         {
# #             "params": torch.zeros(1),
# #             "flops": torch.zeros(1),
# #             "latency": torch.zeros(1),
# #             "macs": torch.zeros(1),
# #         }
# #     )
# # )
