from typing import Callable
from src.torchmodules.nas_modules import MultiOperation, MultiOperationTrackerMixin
from ignite.engine.engine import Engine
import numpy as np


class SupernetGlobalTemperatureUpdate:
    def __init__(
        self,
        model: MultiOperationTrackerMixin,
        initial_value: float,
        update_fn: Callable[[Engine, np.ndarray], None],
    ):
        self.temperature = np.asarray([initial_value])
        self.update_fn = update_fn
        # hook all temperatures to same ndarray object
        mod: MultiOperation
        for mod in model._TrackedMultiOperationModules.values():
            mod.scaling_activation.temperature = self.temperature

    def __call__(self, engine: Engine):
        # save temperature to engine state
        self.update_fn(engine, self.temperature)
        engine.state.temperature = float(self.temperature)


def update_wrapper(update_fn):
    def _f(engine: Engine, temperature):
        n = update_fn(engine.state.iteration)
        temperature[0] = n

    return _f


def create_tanh_decay(
    max: float,
    min: float,
    width: float,
    alpha: float = 3.0,
    rising=False,
):
    """
    Create a tanh decay function that accepts an x-value>=0 and provides corresponding y-value between [min, max].
    """
    h = (max - min) / 2.0
    w = width / 2.0

    def _f(x: np.ndarray):
        a = 1.0 - np.tanh(alpha * (x - w) / w)
        if rising:
            a = 2.0 - a
        return np.maximum(a * h - min, min)

    return _f
