from typing import Callable, List, Mapping
import torch.nn as nn
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
