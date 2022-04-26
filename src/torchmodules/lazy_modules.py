from typing import Callable, List, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from boltons.typeutils import make_sentinel


LAZY_AUTO_CONFIG = make_sentinel("LAZY_AUTO_CONFIG")


class LazyLazyMixin(nn.modules.lazy.LazyModuleMixin):
    def initialize_parameters(self, *args, **kwargs):
        if hasattr(self, "lazy_init_configs"):
            config = self.lazy_init_configs
            for key, default in config.items():
                val = LAZY_AUTO_CONFIG
                if hasattr(self, key):
                    val = getattr(self, key)
                    if callable(val):
                        val = val(self, *args, **kwargs)
                if val is LAZY_AUTO_CONFIG:
                    val = default(self, *args, **kwargs)
                setattr(self, key, val)
        return super().initialize_parameters(*args, **kwargs)


class LazyLazyLinear(LazyLazyMixin, nn.LazyLinear):
    lazy_init_configs: List[str] = {
        "out_features": lambda _, x: x.shape[-1],
    }


class LazyLazyConv2d(LazyLazyMixin, nn.LazyConv2d):
    lazy_init_configs: Mapping[str, Callable] = {
        "out_channels": lambda _, x: x.shape[-3],
    }


class _LazyGroupNormProxy:
    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(input[-3])


class LazyGroupNorm(nn.modules.lazy.LazyModuleMixin, nn.Module):
    def __init__(
        self,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.parameter.UninitializedParameter(**factory_kwargs)
            self.bias = nn.parameter.UninitializedParameter(**factory_kwargs)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def initialize_parameters(self, input) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(input.shape[-3])
                self.bias.materialize(input.shape[-3])
                self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return "{num_groups}, eps={eps}, " "affine={affine}".format(**self.__dict__)
