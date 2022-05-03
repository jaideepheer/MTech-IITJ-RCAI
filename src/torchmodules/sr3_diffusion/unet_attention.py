from functools import partial
import math
from typing import Any, Dict, List
import torch
import torch.nn as nn
from src.torchmodules.activations import GumbelSoftmax, SoftmaxTemperature

from src.torchmodules.lazy_modules import (
    LAZY_AUTO_CONFIG,
    LazyGroupNorm,
    LazyLazyConv2d,
)
from src.torchmodules.mixins import (
    BasicMeasurableMixin,
    BasicMeasurableWrapper,
)
from src.torchmodules.nas_modules import MultiOperation, SupernetMixin
from src.torchmodules.structures import MeasurableSequential, UNetLayer
from src.utils.provider import DictProvider, Provider
import opt_einsum as oe

# =============================================================================================================================
#
# See: See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py
#
# =============================================================================================================================

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(BasicMeasurableMixin, nn.Module):
    def __init__(self, encoding_dim):
        super().__init__()
        count = encoding_dim // 2
        step = torch.arange(count, dtype=torch.float32) / count
        step = step.unsqueeze(0)
        step = torch.exp(-math.log(1e4) * step)
        self.register_buffer("step_multiplier", step)

    def forward(self, noise_level: torch.Tensor):
        encoding = noise_level.unsqueeze(1) * self.step_multiplier
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(BasicMeasurableMixin, nn.Module):
    def __init__(self, n_features, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.LazyLinear(n_features * (1 + self.use_affine_level))
        )

    def forward(self, x: torch.Tensor, *, noise_embedding: torch.Tensor):
        # set kwarg shapes for measurement
        self.set_kwarg_shapes(noise_embedding=noise_embedding.shape)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embedding).view(batch, -1, 1, 1).chunk(2, dim=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embedding).view(batch, -1, 1, 1)
        return x


class GroupNormConvBlock(BasicMeasurableMixin, nn.Module):
    def __init__(self, out_channels, groups=32, dropout=0):
        super().__init__()
        self.block = MeasurableSequential(
            LazyGroupNorm(groups),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            Provider.get(
                "GroupNormConvBuilder",
                default=(lambda c: LazyLazyConv2d(c, 3, padding=1)),
            )(out_channels),
        )

    def forward(self, x):
        return self.block(x)

    def forward_measurements(self, x: Dict[str, torch.Tensor]):
        return self.block.forward_measurements(x)


class ResnetBlock(BasicMeasurableMixin, nn.Module):
    def __init__(
        self,
        out_channels,
        dropout=0,
        use_affine_level=False,
        norm_groups=32,
    ):
        super().__init__()
        self.noise_func = FeatureWiseAffine(out_channels, use_affine_level)

        self.block1 = GroupNormConvBlock(out_channels, groups=norm_groups)
        self.block2 = GroupNormConvBlock(
            out_channels, groups=norm_groups, dropout=dropout
        )
        self.res_conv = BasicMeasurableWrapper.wrap_module(
            nn.LazyConv2d(out_channels, 1)
        )

    def forward(self, x, *, noise_embedding):
        # set kwarg shapes for measurement
        self.set_kwarg_shapes(noise_embedding=noise_embedding.shape)
        h = self.block1(x)
        h = self.noise_func(h, noise_embedding=noise_embedding)
        h = self.block2(h)
        return h + self.res_conv(x)

    def forward_measurements(self, x: Dict[str, torch.Tensor]):
        init = x
        x = self.block1.forward_measurements(x)
        x = self.noise_func.forward_measurements(x)
        x = self.block2.forward_measurements(x)
        # skip connection
        init = self.res_conv.forward_measurements(init)
        x = {k: (x[k] + init[k]) for k in x.keys()}
        return x


class SelfAttention(BasicMeasurableMixin, nn.Module):
    def __init__(self, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = LazyGroupNorm(norm_groups)
        self.qkv = LazyLazyConv2d(lambda _, x: x.shape[-3] * 3, 1, bias=False)
        self.out = LazyLazyConv2d(LAZY_AUTO_CONFIG, 1)

        # eignsum expressions
        self._exp_cache = {}

    def einsum(self, *args, exp_id):
        if exp_id not in self._exp_cache:
            ip_shapes = [x.shape for x in args[1:]]
            self._exp_cache[exp_id] = oe.contract_expression(args[0], *ip_shapes)
        return self._exp_cache[exp_id](*args[1:])

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx",
            query,
            key,
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum(
            "bnhwyx, bncyx -> bnchw",
            attn,
            value,
        ).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(BasicMeasurableMixin, nn.Module):
    def __init__(self, out_channels, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            out_channels, norm_groups=norm_groups, dropout=dropout
        )
        if with_attn:
            self.attn = Provider.get(
                "SelfAttentionBuilder",
                default=(lambda groups: SelfAttention(norm_groups=groups)),
            )(norm_groups)

    def forward(self, x, *, noise_embedding):
        # set kwarg shapes for measurement
        self.set_kwarg_shapes(noise_embedding=noise_embedding.shape)
        x = self.res_block(x, noise_embedding=noise_embedding)
        if self.with_attn:
            x = self.attn(x)
        return x

    def forward_measurements(self, x: Dict[str, torch.Tensor]):
        x = self.res_block.forward_measurements(x)
        if self.with_attn:
            x = self.attn.forward_measurements(x)
        return x


# =============================================================================================================================


class UNetWithAttention(BasicMeasurableMixin, nn.Module):
    def __init__(
        self,
        *,
        out_channels: int,
        inner_channels: int = 32,
        norm_groups: int = 32,
        channel_multipliers: List[int] = [1, 2, 4, 8, 8],
        resnet_blocks_per_unet_layer: int = 3,
        # use attention for layers with image under this resolution
        attention_from_depth: int = 3,
        dropout: float = 0.0,
    ):
        """
        UNet model.
        See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py
        """
        super().__init__()
        self.noise_level_mlp = MeasurableSequential(
            PositionalEncoding(inner_channels),
            nn.Linear(inner_channels, inner_channels * 4),
            nn.SiLU(),
            nn.Linear(inner_channels * 4, inner_channels),
        )
        self.inner_channels = inner_channels
        self.norm_groups = norm_groups
        self.channel_multipliers = channel_multipliers
        self.resnet_blocks_per_unet_layer = resnet_blocks_per_unet_layer
        self.attention_from_depth = attention_from_depth
        self.dropout = dropout

        self.head = BasicMeasurableWrapper.wrap_module(
            nn.LazyConv2d(inner_channels, kernel_size=3, padding=1)
        )
        # make unet layers
        self.net = self._make_unet_from_depth()
        self.tail = GroupNormConvBlock(out_channels, groups=norm_groups)

    def _make_lowest_layer(self):
        return MeasurableSequential(
            ResnetBlocWithAttn(
                self.inner_channels * self.channel_multipliers[-1],
                norm_groups=self.norm_groups,
                dropout=self.dropout,
                with_attn=True,
            ),
            ResnetBlocWithAttn(
                self.inner_channels * self.channel_multipliers[-1],
                norm_groups=self.norm_groups,
                dropout=self.dropout,
                with_attn=False,
            ),
        )

    def _make_resnet_layers(self, subnet, depth, resnet_idx=None):
        if resnet_idx is None:
            resnet_idx = self.resnet_blocks_per_unet_layer
        down_ch = self.channel_multipliers[depth] * self.inner_channels
        up_ch = self.channel_multipliers[-(depth + 1)] * self.inner_channels
        if resnet_idx == 0:
            # all resnet blocks for layer created
            return subnet
        else:
            return self._make_resnet_layers(
                UNetLayer(
                    down=ResnetBlocWithAttn(
                        down_ch,
                        norm_groups=self.norm_groups,
                        dropout=self.dropout,
                        with_attn=(depth >= self.attention_from_depth),
                    ),
                    subnet=subnet,
                    up=ResnetBlocWithAttn(
                        up_ch,
                        norm_groups=self.norm_groups,
                        dropout=self.dropout,
                        with_attn=(depth >= self.attention_from_depth),
                    ),
                ),
                depth,
                resnet_idx=(resnet_idx - 1),
            )

    def _make_downscale_module(self, scale):
        # assert scale is power of 2
        if scale and (not (scale & (scale - 1))):
            return MeasurableSequential(
                LazyLazyConv2d(LAZY_AUTO_CONFIG, 3, scale, 1),
                ignore_forward_kwargs=True,
            )
        else:
            raise ValueError(
                f"Downscalling can only be done for scales of pow of 2. scale={scale}"
            )

    def _make_scaling_layers(self, subnet, scaling=2):
        return UNetLayer(
            # downsample
            down=self._make_downscale_module(scaling),
            subnet=subnet,
            # upsample
            up=MeasurableSequential(
                nn.Upsample(scale_factor=scaling, mode="nearest"),
                LazyLazyConv2d(LAZY_AUTO_CONFIG, 3, padding=1),
                ignore_forward_kwargs=True,
            ),
        )

    def _make_unet_from_depth(self, depth=0):
        if depth == len(self.channel_multipliers):
            # this is the lowest base flat layer
            return self._make_lowest_layer()
        else:
            return self._make_resnet_layers(
                self._make_scaling_layers(
                    self._make_unet_from_depth(depth + 1),
                ),
                depth,
            )

    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        noise = self.noise_level_mlp(noise)
        x = self.head(x)
        x = self.net(x, noise_embedding=noise)
        x = self.tail(x)
        return x

    def forward_measurements(self, x: Dict[str, torch.Tensor]):
        x = self.noise_level_mlp.forward_measurements(x)
        x = self.head.forward_measurements(x)
        x = self.net.forward_measurements(x)
        x = self.tail.forward_measurements(x)
        return x


# =============================================================================================================================

_default_conv_supernet_space = [
    {
        "kernel_size": 1,
        "stride": 1,
    },
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

_activation_builders = {
    "softmax": lambda: nn.Softmax(dim=0),
    "softmax_temperature": lambda: SoftmaxTemperature(temperature=20.0, dim=0),
    "gunbel_softmax": lambda: GumbelSoftmax(temperature=20.0, dim=0),
}


def _supernet_conv_builder(
    out_channels: int,
    supernet_space: List[Dict[str, Any]] = _default_conv_supernet_space,
):
    return MultiOperation(
        [LazyLazyConv2d(out_channels, **i) for i in supernet_space],
        scaling=Provider.get("scaling", None),
        scaling_activation=Provider.get(
            "scaling_activation",
            default_factory=_activation_builders["softmax_temperature"],
        ),
        is_scaling_learnable=True,
        execution_mode="sequential",
    )


def _supernet_self_attention_builder(groups):
    return MultiOperation(
        [
            SelfAttention(norm_groups=groups),
            GroupNormConvBlock(LAZY_AUTO_CONFIG, groups=groups),
        ]
        + [BasicMeasurableWrapper.wrap_module(nn.Identity())],
        scaling=Provider.get("scaling", None),
        scaling_activation=Provider.get(
            "scaling_activation",
            default_factory=_activation_builders["softmax"],
        ),
        is_scaling_learnable=True,
        execution_mode="sequential",
    )


class UNetWithAttentionSupernet(SupernetMixin, UNetWithAttention):
    def __init__(
        self,
        *,
        out_channels: int,
        inner_channels: int = 32,
        norm_groups: int = 32,
        channel_multipliers: List[int] = [1, 2, 4, 8, 8],
        resnet_blocks_per_unet_layer: int = 3,
        dropout: float = 0.0,
        # supernet args
        conv_search_space: List[Dict[str, Any]] = _default_conv_supernet_space,
        # use attention for layers under this depth
        attention_from_depth: int = 3,
        **kwargs,
    ):
        root_provider = DictProvider(
            {
                "GroupNormConvBuilder": partial(
                    _supernet_conv_builder,
                    supernet_space=conv_search_space,
                ),
                "SelfAttentionBuilder": _supernet_self_attention_builder,
            }
        )
        with root_provider:
            super().__init__(
                out_channels=out_channels,
                inner_channels=inner_channels,
                norm_groups=norm_groups,
                channel_multipliers=channel_multipliers,
                resnet_blocks_per_unet_layer=resnet_blocks_per_unet_layer,
                attention_from_depth=attention_from_depth,
                dropout=dropout,
                **kwargs,
            )
