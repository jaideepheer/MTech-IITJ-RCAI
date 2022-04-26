from typing import Dict, Optional
from collections import OrderedDict
import torch.nn as nn
import torch
from src.torchmodules.mixins import (
    BasicMeasurableMixin,
    BasicMeasurableWrapper,
    MeasurableModuleMixin,
)
from botorch.utils.torch import BufferDict


class MeasurableSequential(BasicMeasurableMixin, nn.Sequential):
    def __init__(
        self,
        *_,
        wrap_non_measurable: bool = True,
        ignore_forward_kwargs: bool = False,
        **kwargs,
    ):
        """
        A nn.Sequential with `forward_measurements` support.

        It also supports `kwargs` in the `forward` method which are passed to every module in a FiLM fashion.
        The `kwargs` are not propagated sequentially, i.e. they are passed to every module call directly.

        Parameters
        ----------
        wrap_non_measurable: bool = True
            If True, all modules that do not inherit MeasurableModuleMixin will be wrapped using BasicMeasurableWrapper.
        ignore_forward_kwargs: bool = False
            If True, ignores kwargs passed to `forward`.

        Raises
        ------
        ValueError
            If a child does not inherit MeasurableModuleMixin and wrap_non_measurable=False.
        """
        self.wrap_non_measurable = wrap_non_measurable
        self.ignore_forward_kwargs = ignore_forward_kwargs
        super().__init__(*_, **kwargs)
        self.seqmodules = list(
            filter(
                lambda x: (
                    not isinstance(
                        x,
                        (
                            BufferDict,
                            nn.ParameterDict,
                            nn.ParameterList,
                            nn.ModuleDict,
                            nn.ModuleList,
                        ),
                    )
                ),
                self,
            )
        )

    def add_module(self, name: str, module: Optional[nn.Module]) -> None:
        if not isinstance(module, MeasurableModuleMixin):
            if self.wrap_non_measurable is False:
                raise ValueError(
                    f"MeasurableSequential requires all children to inherit MeasurableModuleMixin. {module.__class__} is invalid."
                )
            module = BasicMeasurableWrapper(module=module, model_name=name)
        return super().add_module(name, module)

    def forward_measurements(
        self, measurements: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        rt = measurements
        for m in self.seqmodules:
            rt = m.forward_measurements(rt)
        return rt

    def forward(self, x, **kwargs):
        if self.ignore_forward_kwargs is True:
            kwargs = {}
        for m in self.seqmodules:
            x = m(x, **kwargs)
        return x


class MeasurableSkipConnect(MeasurableSequential):
    def __init__(
        self,
        *_,
        skip_connect_measurements: bool = True,
        skip_connect_scaling: float = 1.0,
        skip_connect_concat_dim: int = None,
        **kwargs,
    ):
        """
        Extends nn.Sequential to add a skip connection over the entire sequence.

        Parameters
        ----------
        skip_connect_measurements : bool, optional
            If True, `forward_measurements` adds measurements from skip connect to the end result.
            Else the skip connection is ignored for `forward_measurements`.
            By default True.
        skip_connect_scaling: float = 1.0
            Scaling for the skip connection.
        """
        super().__init__(*_, **kwargs)
        self.skip_connect_measurements = skip_connect_measurements
        self.skip_connect_scaling = skip_connect_scaling
        self.skip_connect_concat_dim = skip_connect_concat_dim

    def forward_measurements(
        self, measurements: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        rt = super().forward_measurements(measurements)
        if self.skip_connect_measurements is True:
            # add skip connect metrics
            rt = {
                k: (rt[k] + (measurements[k] * self.skip_connect_scaling))
                for k in rt.keys()
            }
        return rt

    def forward(self, x, **kw):
        rt = super().forward(x, **kw)
        if self.skip_connect_concat_dim is None:
            rt = rt + (x * self.skip_connect_scaling)
        else:
            rt = torch.cat(
                [rt, x * self.skip_connect_scaling],
                dim=self.skip_connect_concat_dim,
            )
        return rt


class UNetLayer(BasicMeasurableMixin):
    def __init__(
        self,
        *_,
        down: nn.Module,
        subnet: nn.Module,
        up: nn.Module,
        skip_connect_scaling: float = 1.0 / (2**0.5),
        skip_connect_measurements: bool = True,
        skip_connect_concat_dim: int = -3,
        ignore_forward_kwargs: bool = False,
    ):
        super().__init__()
        self.ignore_forward_kwargs = ignore_forward_kwargs
        self.module = MeasurableSequential(
            OrderedDict(
                [
                    ("unet_layer_down", down),
                    (
                        "unet_layer_child",
                        MeasurableSkipConnect(
                            subnet,
                            skip_connect_scaling=skip_connect_scaling,
                            skip_connect_measurements=skip_connect_measurements,
                            skip_connect_concat_dim=skip_connect_concat_dim,
                        ),
                    ),
                    ("unet_layer_up", up),
                ]
            )
        )

    def forward(self, x, **kwargs):
        if self.ignore_forward_kwargs:
            kwargs = {}
        return self.module(x, **kwargs)

    def forward_measurements(
        self, measurements: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.module.forward_measurements(measurements)


# TODO REPL remove when done

# from rich import print
# from torchinfo import summary

# m = UNetLayer(
#     pre_skip=nn.Linear(2, 4),
#     child=nn.Linear(4, 4),
#     post_skip=nn.Linear(4, 2),
#     skip_connect_scaling=1.0,
#     skip_connect_measurements=False,
# )
# m = m.to("cuda")
# print(m)

# ip = torch.ones((2,), device="cuda")
# print("op:", m(ip))

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

# print(
#     m.forward_measurements(
#         {
#             "params": torch.zeros(1, device="cuda"),
#             "flops": torch.zeros(1, device="cuda"),
#             "latency": torch.zeros(1, device="cuda"),
#             "macs": torch.zeros(1, device="cuda"),
#         }
#     )
# )
