from itertools import chain
import logging
from typing import Callable, Dict, Literal, Set, Tuple, Union
from boltons.iterutils import first
import torch
import torch.nn as nn
from botorch.utils.torch import BufferDict
from deepspeed.profiling.flops_profiler.profiler import (
    FlopsProfiler,
    wrapFunc,
    _einsum_flops_compute,
)
import deepspeed.profiling.flops_profiler.profiler as profiler
from morecontext import attrset

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# ================================================
# Bugfix opt_einsum and deepspeed compatability
# ================================================
_original_einsum = torch.einsum


def _fix_eignsum(equation, *operands):
    if len(operands) > 0 and isinstance(operands[0], tuple):
        operands = operands[0]
    return _einsum_flops_compute(equation, *operands)


def _fix_patch_methods(original):
    def _f():
        global _original_einsum
        _original_einsum = torch.einsum
        original()
        torch.einsum = wrapFunc(_original_einsum, _fix_eignsum)

    return _f


def _fix_reload_methods(original):
    def _f():
        global _original_einsum
        original()
        torch.einsum = _original_einsum

    return _f


profiler._patch_tensor_methods = _fix_patch_methods(profiler._patch_tensor_methods)
profiler._reload_tensor_methods = _fix_reload_methods(profiler._reload_tensor_methods)

# ================================================


class ModelBuilderMixin(nn.Module):
    def __init__(
        self,
        *_,
        module: nn.Module = None,
        module_builder: Callable[[], nn.Module] = None,
        model_name: str = None,
        **kwargs,
    ) -> None:
        """
        Adds support setting a pre-instantiated model or using a model builder. Sets the model to `self.model`.

        Parameters
        ----------
        module : nn.Module, optional
            A `nn.Module` instance. By default None
        module_builder : Callable[[], nn.Module], optional
            A function that returns a the torch module. By default None

        Raises
        ------
        Exception
            Raised if none of or both `module_builder` and/or `module` are provided.
        """
        if not ((module is None) ^ (module_builder is None)):
            raise Exception("Exactly one of module or module_builder must be passed.")
        # init. everything before making model
        super().__init__(*_, **kwargs)
        # make model
        if module is not None:
            if not isinstance(module, nn.Module):
                raise ValueError("Passed module not nn.Module.", module)
            self.module = module
        else:
            self.module = module_builder()
        # save name
        self.model_name = model_name


class ShapeRecorderMixin(nn.Module):
    def __init__(self, *_, **kwargs):
        super().__init__(*_, **kwargs)
        self.clear_shape_recording()
        self._recorded_input_shapes = None
        self._recorded_output_shapes = None
        self._recorded_kwargs_shapes = None

    def _shape_recorder_forward(self, module, inputs, outputs):
        # remove hook
        self._shape_recorder_hook.remove()
        # record shapes
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        self._recorded_input_shapes = tuple([tuple(x.shape) for x in inputs])
        self._recorded_output_shapes = tuple([tuple(x.shape) for x in outputs])

    def input_shapes(self) -> Tuple[Tuple, ...]:
        """
        Returns a tuple of shapes for each non-kwarg input tensor.

        Returns
        -------
        Tuple[Tuple, ...]
            A tuple of shapes for each non-kwarg input tensor

        Raises
        ------
        Exception
            If called before calling forward.
        """
        if self._recorded_input_shapes is None:
            raise Exception(
                "Please call forward at least once before calling input_shapes."
            )
        return self._recorded_input_shapes

    def kwarg_shapes(self):
        if self._recorded_kwargs_shapes is None:
            return {}
        return self._recorded_kwargs_shapes

    def set_kwarg_shapes(self, **kwargs):
        self._recorded_kwargs_shapes = kwargs

    def output_shapes(self) -> Tuple[Tuple, ...]:
        """
        Returns a tuple of shapes for each non-kwarg output tensor.

        Returns
        -------
        Tuple[Tuple, ...]
            A tuple of shapes for each non-kwarg output tensor

        Raises
        ------
        Exception
            If called before calling forward.
        """
        if self._recorded_output_shapes is None:
            raise Exception(
                "Please call forward at least once before calling output_shapes."
            )
        return self._recorded_output_shapes

    def clear_shape_recording(self):
        """
        Clear recorded shapes and re-records on next forward call.
        """
        self._shape_recorder_hook = self.register_forward_hook(
            self._shape_recorder_forward
        )
        self._recorded_input_shapes = None
        self._recorded_output_shapes = None
        self._recorded_kwargs_shapes = None


class MeasurableModuleMixin(nn.Module):
    def __init__(self, *_, **kwargs):
        """
        Adds metric measurement capabilities to a module.
        By default all measurements are cached once measured.

        Use the `clear` measurement method to clear some/all measurements.
        """
        super().__init__(*_, **kwargs)
        self._measurements_cache = BufferDict()

    def clear(self, metrics: Union[Literal["all"], Set[str]]):
        if metrics == "all":
            self._measurements_cache.clear()
        else:
            for k in metrics:
                if k in self._measurements_cache:
                    del self._measurements_cache[k]

    @torch.no_grad()
    def _run_measurement(self, metrics: Set[str]) -> Dict[str, torch.Tensor]:
        """
        Override this function to add support of further measurements.
        Do not perform caching in this method, this method's return value is automatically cached.

        Parameters
        ----------
        metrics: Set[str]
            The names of the measurement metrics to run a measurement for.
            These must be small case.

        Returns
        -------
        dict
            A dict mapping each metric name to the measurement value.
            The measurement values will be passed to torch.as_tensor so ensure their type is compatable.
            If you cannot implement a metric measurement, simply call super().run_measurement() with the remaining metrics and update the return dict.
        """
        return {}

    def get_measurements(
        self, metrics: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Retruns the cached measurements for the requested metrics.
        Call this method to retrieve measurements for the requested metrics.

        Parameters
        ----------
        metrics : Set[str]
            The names of the measurement metrics. This is case insensitive.

        Returns
        -------
        dict
            A dict mapping each metric name to the measurement torch.Tensor value.

        Raises
        ------
        Exception
            If no measurement for a requested metric is available.
        """
        LOG.debug(
            f"get_measurements called: name={getattr(self, 'model_name', None)} cls={self.__class__}"
        )
        # lower case metrics
        metrics = {x.lower(): v for x, v in metrics.items()}
        # filter cached metrics
        to_measure = {
            x: v for x, v in metrics.items() if x not in self._measurements_cache
        }
        if len(to_measure) > 0:
            with torch.no_grad():
                measurements = self._run_measurement(to_measure)
                # convert measurements to tensors
                device = first(map(lambda x: x.device, self.parameters()))
                measurements = {
                    k: (
                        v.to(device)
                        if torch.torch.is_tensor(v)
                        else torch.as_tensor(
                            v,
                            device=device,
                        )
                    )
                    for k, v in measurements.items()
                }
                # update cached metrics
                self._measurements_cache.update(measurements)
            # check if all measurements available
            if not set(to_measure.keys()).issubset(measurements.keys()):
                raise Exception(
                    f"Not all requested measurements could be measured, requested ({to_measure}), measured ({list(measurements.keys())})"
                )
        # return dict of requested measurements
        return {k: self._measurements_cache[k] for k in metrics}

    def forward_measurements(
        self, measurements: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Given a dict of str and torch.Tensor from previous module measurements, returns this module's measurements merged with the previous ones'.
        The returned dict must be a copy of the passed dict.

        Default behavior is to simply add the values in the passed dict to the values returned by `self.get_measurements`.

        Parameters
        ----------
        measurements: Dict[str, torch.Tensor]
            Immutable input measurements dict.

        Returns
        -------
        Dict[str, torch.Tensor]
            A new dict of combined measurements.
        """
        LOG.debug(
            f"default forward_measurements called: name={getattr(self, 'model_name', None)} cls={self.__class__}"
        )
        metrics = self.get_measurements(measurements)
        return {k: (metrics[k] + measurements[k]) for k in metrics.keys()}


class BasicMeasurableMixin(ShapeRecorderMixin, MeasurableModuleMixin):
    """
    Adds 'flops', 'macs', 'latency' and 'params' measurements using deepspeed profiler.
    """

    @torch.no_grad()
    def _run_measurement(self, metrics: Dict[str, torch.Tensor]):
        ops = {"flops", "macs", "latency", "params"}
        # calculatable metrics
        metrics = {k: v for k, v in metrics.items() if k in ops}
        rt = {}
        # check for unsupported metrics
        if len(metrics) > 0:
            # create ip args
            device = next(iter(metrics.values())).device
            dtype = next(iter(metrics.values())).dtype
            args = [
                torch.zeros(x, device=device, dtype=dtype) for x in self.input_shapes()
            ]
            kwargs = {
                k: torch.zeros(x, device=device, dtype=dtype)
                for k, x in self.kwarg_shapes().items()
            }
            # profile
            with torch.no_grad(), attrset(self, "train", False):
                # ===============================
                # patch cuda synchronize method
                # ===============================
                original_sync = torch.cuda.synchronize

                def _f():
                    original_sync(device)

                torch.cuda.synchronize = _f
                # ===============================
                profiler = FlopsProfiler(self)
                LOG.debug(
                    f"Starting profile: name={getattr(self, 'model_name', None)}, cls={self.__class__}"
                )
                profiler.start_profile()
                self(*args, **kwargs)
                # calculate metrics
                flops = profiler.get_total_flops()
                macs = profiler.get_total_macs()
                params = profiler.get_total_params()
                latency = profiler.get_total_duration()
                profiler.stop_profile()
                # ===============================
                # unpatch cuda synchronize method
                # ===============================
                torch.cuda.synchronize = original_sync
                # ===============================
                mt = {
                    "flops": flops,
                    "macs": macs,
                    "params": params,
                    "latency": latency,
                }
                LOG.debug(
                    f"Profile done: name={getattr(self, 'model_name', None)}, cls={self.__class__}, metrics={mt}"
                )
            rt.update(mt)
        # pass other metrics to super
        rt.update(super()._run_measurement(metrics))
        return rt


class BasicMeasurableWrapper(BasicMeasurableMixin, ModelBuilderMixin):
    @staticmethod
    def wrap_module(module: nn.Module, force: bool = False):
        """
        Wraps module in BasicMeasurableWrapper if needed.

        Parameters
        ----------
        module : nn.Module
            The module to wrap.
        force : bool, optional
            If True will wrap module even if it inherits `MeasurableModuleMixin`, by default True
        """
        if not isinstance(module, MeasurableModuleMixin) or (force is True):
            module = BasicMeasurableWrapper(module=module)
        return module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
