import math
from typing import List
import torch
import torch.nn as nn

from src.torchmodules.lazy_modules import (
    LAZY_AUTO_CONFIG,
    LazyGroupNorm,
    LazyLazyConv2d,
)
from src.torchmodules.structures import MeasurableSequential, UNetLayer

# =============================================================================================================================
#
# See: See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py
#
# =============================================================================================================================

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim

    def forward(self, noise_level: torch.Tensor):
        count = self.encoding_dim // 2
        step = (
            torch.arange(count, dtype=noise_level.dtype, device=noise_level.device)
            / count
        )
        encoding = noise_level.unsqueeze(1) * torch.exp(
            -math.log(1e4) * step.unsqueeze(0)
        )
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, n_features, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.LazyLinear(n_features * (1 + self.use_affine_level))
        )

    def forward(self, x: torch.Tensor, *, noise_embedding: torch.Tensor):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = (
                self.noise_func(noise_embedding).view(batch, -1, 1, 1).chunk(2, dim=1)
            )
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embedding).view(batch, -1, 1, 1)
        return x


class GroupNormConvBlock(nn.Module):
    def __init__(self, out_channels, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            LazyGroupNorm(groups),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.LazyConv2d(out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
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
        self.res_conv = nn.LazyConv2d(out_channels, 1)

    def forward(self, x, *, noise_embedding):
        h = self.block1(x)
        h = self.noise_func(h, noise_embedding=noise_embedding)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = LazyGroupNorm(norm_groups)
        self.qkv = LazyLazyConv2d(lambda _, x: x.shape[-3] * 3, 1, bias=False)
        self.out = LazyLazyConv2d(LAZY_AUTO_CONFIG, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, out_channels, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            out_channels, norm_groups=norm_groups, dropout=dropout
        )
        if with_attn:
            self.attn = SelfAttention(norm_groups=norm_groups)

    def forward(self, x, *, noise_embedding):
        x = self.res_block(x, noise_embedding=noise_embedding)
        if self.with_attn:
            x = self.attn(x)
        return x


# =============================================================================================================================


class UNetWithAttention(nn.Module):
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
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channels),
            nn.Linear(inner_channels, inner_channels * 4),
            nn.SiLU(),
            nn.Linear(inner_channels * 4, inner_channels),
        )

        def _make_unet(depth=0, resnet_idx=resnet_blocks_per_unet_layer):
            if depth == len(channel_multipliers):
                # this is the lowest base flat layer
                return MeasurableSequential(
                    ResnetBlocWithAttn(
                        inner_channels * channel_multipliers[-1],
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=True,
                    ),
                    ResnetBlocWithAttn(
                        inner_channels * channel_multipliers[-1],
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=False,
                    ),
                )
            else:
                down_ch = channel_multipliers[depth] * inner_channels
                up_ch = channel_multipliers[-(depth + 1)] * inner_channels
                if resnet_idx == 0:
                    # all resnet blocks for layer created, this should be scaling layer
                    # reset recursion stats
                    resnet_idx = resnet_blocks_per_unet_layer
                    depth += 1
                    return UNetLayer(
                        # downsample
                        down=MeasurableSequential(
                            LazyLazyConv2d(LAZY_AUTO_CONFIG, 3, 2, 1),
                            ignore_forward_kwargs=True,
                        ),
                        subnet=_make_unet(depth, resnet_idx),
                        # upsample
                        up=MeasurableSequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            LazyLazyConv2d(LAZY_AUTO_CONFIG, 3, padding=1),
                            ignore_forward_kwargs=True,
                        ),
                    )
                else:
                    # need more resnet blocks for this layer
                    resnet_idx -= 1
                    return UNetLayer(
                        down=ResnetBlocWithAttn(
                            down_ch,
                            norm_groups=norm_groups,
                            dropout=dropout,
                            with_attn=(depth >= attention_from_depth),
                        ),
                        subnet=_make_unet(depth, resnet_idx),
                        up=ResnetBlocWithAttn(
                            up_ch,
                            norm_groups=norm_groups,
                            dropout=dropout,
                            with_attn=(depth >= attention_from_depth),
                        ),
                    )

        self.head = nn.LazyConv2d(inner_channels, kernel_size=3, padding=1)
        # make unet layers
        self.net = _make_unet()
        self.tail = GroupNormConvBlock(out_channels, groups=norm_groups)

    def forward(self, x: torch.Tensor, noise: torch.Tensor):
        noise = self.noise_level_mlp(noise)
        x = self.head(x)
        x = self.net(x, noise_embedding=noise)
        x = self.tail(x)
        return x
