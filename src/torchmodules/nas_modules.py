from contextlib import contextmanager
from itertools import chain
import logging
from typing import Any, Callable, Dict, List, Union, Literal, Tuple
import torch
import torch.nn as nn
import torch.fx as fx
from collections import Counter
from deprecated import deprecated
from src.models.torch_modules.structures import MeasurableSequential
import src.utils.provider as P
from frozendict import frozendict
from boltons.iterutils import first
from botorch.utils.torch import BufferDict
import opt_einsum as oe

from src.models.torch_modules.mixins import (
    BasicMeasurableMixin,
    BasicMeasurableWrapper,
    ModelBuilderMixin,
)
from src.utils.utils import get_logger

LOG = get_logger(__name__)
LOG.setLevel(logging.INFO)


class MultiOperation(BasicMeasurableMixin, nn.Module):
    def __init__(
        self,
        operations: List[nn.Module],
        scaling: Union[List[float], nn.Parameter] = None,
        scaling_activation: nn.Module = nn.Softmax(dim=0),
        is_scaling_learnable: bool = True,
        execution_mode: Literal["sequential", "parallel", "jit"] = "sequential",
    ):
        """
        Performs multiple operations on the same input data adds results.
        This also provides masking features to disable operations using a torch.BoolTensor mask.

        Parameters
        ----------
        operations : List[nn.Module]
            The operations to perform on given input.
        scaling: Union[List[float], torch.Tensor, nn.Parameter] = None
            The scaling factors to apply to each operation's output.
            If `List[float]` then `is_scaling_learnable` is used to store scaling as a Parameter or a Buffer.
            If `nn.Parameter` then the passed value is stored as is. This is usefull in reusing the same `nn.Parameter` objects.
        scaling_activation: nn.Module = nn.Softmax()
            The activation function to applied to the scaling tensor before operation outputs are scaled.
        """
        super().__init__()
        assert len(operations) > 1
        assert not (
            isinstance(scaling, nn.Parameter) and (is_scaling_learnable is False)
        )
        # save scaling
        if scaling is None:
            scaling = [1] * len(operations)
        if isinstance(scaling, nn.Parameter):
            self.MultiOperation_scaling = scaling
        else:
            params = chain(*map(lambda x: x.parameters(), operations))
            device = first(map(lambda x: x.device, params))
            dtype = first(map(lambda x: x.dtype, params), default=torch.float)
            scaling = torch.as_tensor(scaling, dtype=dtype, device=device)
            if is_scaling_learnable is True:
                self.register_parameter("MultiOperation_scaling", nn.Parameter(scaling))
            else:
                self.register_buffer("MultiOperation_scaling", scaling)
            if scaling is None:
                scaling = [1] * len(operations)
        # assert shapes
        assert len(operations) == len(scaling)
        self.operations = nn.ModuleList(operations)
        self.scaling_activation = scaling_activation
        # select forward function
        if execution_mode == "sequential":
            self.__forward_function = self.__sequential_forward
        elif execution_mode == "parallel":
            self.__forward_function = self.__parallel_forward
        else:
            self.__forward_function = self.__jit_forward
        self.execution_mode = execution_mode
        # create mask
        self.register_buffer(
            "operation_mask",
            torch.ones(
                len(operations),
                dtype=torch.bool,
                device=scaling.device,
            ),
        )

    def maskIdx_and_activatedScaling(
        self,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Returns two tensors, a list of active operation indices and the corresponding activated scaling tensor.

        Returns
        -------
        torch.LongTensor
            A tensor of indices of active operations.
        torch.FloatTensor
            A tensor of activated scaling values per operation.
        """
        maskIdx = self.operation_mask.nonzero().squeeze(dim=1)
        scaling = self.MultiOperation_scaling[maskIdx]
        scaling = self.scaling_activation(scaling)
        return maskIdx, scaling

    def __sequential_forward(
        self,
        scaling: torch.Tensor,
        operations: List[nn.Module],
        args: torch.Tensor,
    ):
        rt = 0
        for i, op in enumerate(operations):
            rt = rt + (op(args) * scaling[i])
        return rt

    def __parallel_forward(
        self,
        scaling: torch.Tensor,
        operations: List[nn.Module],
        args: torch.Tensor,
    ):
        outputs = torch.stack([op(args) for op in operations])
        return oe.contract("o,o...->...", scaling, outputs)

    def __jit_forward(
        self,
        scaling: torch.Tensor,
        operations: List[nn.Module],
        args: torch.Tensor,
    ):
        outputs = [torch.jit.fork(op, args) for op in operations]
        outputs = torch.stack([torch.jit.wait(m) for m in outputs])
        return oe.contract("o,o...->...", scaling, outputs)

    def forward(self, args):
        if hasattr(self, "__flops__"):
            raise Exception("MultiOperation profiled outside forward_measurement.")
        # mask operations and scaling
        mask_idx, scaling = self.maskIdx_and_activatedScaling()
        operations = [self.operations[i] for i in mask_idx]
        # perform operations
        rt = self.__forward_function(scaling, operations, args)
        # single/multi return values
        return rt

    def forward_measurements(
        self, measurements: frozendict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        `forward_measurements` is treated specially since each measurement is scaled using `self.scaling`.
        """
        LOG.debug(
            f"MultiOperation.forward_measurements called: name={getattr(self, 'model_name', None)}"
        )
        maskIdx, scaling = self.maskIdx_and_activatedScaling()
        operations = [self.operations[i] for i in maskIdx]
        rt = {}
        # accumulate measurements
        for op in operations:
            m = op.forward_measurements(measurements)
            for k, v in m.items():
                if k not in rt:
                    rt[k] = []
                rt[k].append(v)
        # stack measurements
        return {
            k: oe.contract("o,o...->...", scaling, torch.stack(v, dim=0))
            for k, v in rt.items()
        }


@deprecated(
    reason="Don't use LazyLearnableScaler. Use MultiOperation instead.", action="error"
)
class LazyLearnableScalerTrackerMixin(ModelBuilderMixin):
    def __init__(self, *_, **kwargs):
        self.LazyLearnableScaler_hitcounts = Counter()
        self.LazyLearnableScaler_pathcounts = Counter()
        scalers = {}

        # this captures and stores all LazyLearnableScaler instances fetched using `Provider.get`.
        def _onKey(value, key, found, fullpath, provider):
            name = value.__class__.__qualname__
            if name.endswith("LazyLearnableScaler"):
                if value not in self.LazyLearnableScaler_hitcounts:
                    # add values in seen order
                    scalers[fullpath] = value
                self.LazyLearnableScaler_hitcounts[value] += 1
                self.LazyLearnableScaler_pathcounts[fullpath] += 1

        # call super with provider context
        with P.EventListener("key.returned", _onKey):
            super().__init__(*_, **kwargs)

        if len(scalers) == 0:
            raise ValueError("Supernet has no LazyLearnableScaler instances.")

        # store scalers
        self.LazyLearnableScalers = nn.ModuleDict(scalers)


class MultiOperationTrackerMixin(ModelBuilderMixin):
    def __init__(self, *_, **kwargs):
        super().__init__(*_, **kwargs)
        self._TrackedMultiOperationModules: nn.ModuleDict[
            str, "MultiOperation"
        ] = nn.ModuleDict(
            {
                k.replace(".", "/"): v
                for k, v in self.module.named_modules()
                if (v._get_name() == "MultiOperation") or isinstance(v, MultiOperation)
            }
        )
        if len(self._TrackedMultiOperationModules) == 0:
            raise ValueError("Supernet has no MultiOperation instances.")


class SupernetMixin(MultiOperationTrackerMixin):
    def __init__(
        self,
        *_,
        subnet_mask_fn: Callable[[Dict[str, MultiOperation]], None] = None,
        **kwargs,
    ):
        """
        Allows treating `self.module` as a supernet and provides functions for masking and extracting subnets.

        Parameters
        ----------
        subnet_mask_fn : Callable[[Dict[str, MultiOperation]], None], optional
            This function updates MultiOperation masks and returns nothing.
            If `None`, it uses the argmax of activated scaling tensor` as the only `True` value in its mask.
        """
        super().__init__(*_, **kwargs)
        self.subnet_mask_fn = subnet_mask_fn or self.default_subnet_mask_fn
        self.__subnet_mask_counter = 0

    @property
    def is_subnet_masked(self):
        return self.__subnet_mask_counter > 0

    @contextmanager
    def subnet_masked(self):
        """
        Context manager that applies subnet masks and reverts to previous masks on exit.
        """
        with torch.no_grad():
            operations: Dict[str, MultiOperation] = self._TrackedMultiOperationModules
            # backup masks
            backup = {k: v.operation_mask.clone().detach() for k, v in operations.items()}
            # update masks
            self.subnet_mask_fn(operations)
            self.__subnet_mask_counter += 1
        yield
        with torch.no_grad():
            # restore backup masks
            for k, v in operations.items():
                v.operation_mask = backup[k]
            self.__subnet_mask_counter -= 1

    def default_subnet_mask_fn(self, operations: Dict[str, MultiOperation]):
        for path, op in operations.items():
            maskIx, scaling = op.maskIdx_and_activatedScaling()
            finalIdx = maskIx[scaling.argmax()]
            op.operation_mask.fill_(False)
            op.operation_mask[finalIdx] = True

    @property
    def current_subnet(self) -> fx.graph_module.GraphModule:
        """
        Returns a `fx.GraphModule` traced with the subnet masks applied.
        This module can then be used to store the subnet since it only references the subnet's parameters.
        """
        with self.subnet_masked():
            subnet = fx.symbolic_trace(self.module)
            subnet.load_state_dict(self.module.state_dict())
            return subnet


# TODO AREPL in progress, remove when done
# from rich import print

# m = P.DictProvider({"scaling": P.PathCachedDefaultValue()})(
#     lambda: MultiOperation(
#         [
#             MultiOperation(
#                 [
#                     BasicMeasurableWrapper(
#                         module=nn.Identity(),
#                     )
#                     for _ in range(2)
#                 ],
#                 scaling=P.Provider.get(
#                     "scaling",
#                     default_factory=lambda: nn.Parameter(torch.ones(2)),
#                 ),
#                 execution_mode="sequential",
#                 is_scaling_learnable=True,
#             )
#             for _ in range(2)
#         ],
#         scaling=P.Provider.get(
#             "scaling",
#             default_factory=lambda: nn.Parameter(torch.ones(2)),
#         ),
#         execution_mode="sequential",
#         is_scaling_learnable=True,
#     )
# )()
# m = m.to("cuda")
# # print(dict(m.named_parameters()))
# # print(set(map(lambda x: (x[1], x[0]), m.named_parameters())))

# pp = {n: op for n, op in m.named_parameters() if n.endswith("MultiOperation_scaling")}
# print(pp)


# m = MultiOperation(
#     [
#         BasicMeasurableWrapper.wrap_module(nn.Linear(4, 1)),
#         BasicMeasurableWrapper.wrap_module(nn.Linear(4, 1)),
#     ],
#     execution_mode="jit",
# )
# print([x.shape for x in m.maskIdx_and_activatedScaling()])

# ip = torch.ones((4,))
# print("op:", m(ip))
# print(m.forward_measurements({"flops": torch.zeros(1)}))

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
#             "params": torch.zeros((1), device="cuda"),
#             "flops": torch.zeros((1), device="cuda"),
#             "latency": torch.zeros((1), device="cuda"),
#             "macs": torch.zeros((1), device="cuda"),
#         }
#     )
# )

# print("done")
