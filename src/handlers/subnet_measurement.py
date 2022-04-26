from typing import Dict, Union
from ignite.engine.engine import Engine
import torch
from src.torchmodules.mixins import MeasurableModuleMixin
from src.torchmodules.nas_modules import SupernetMixin
from src.utils.utils import get_module_device


class SubnetMeasurement:
    def __init__(
        self,
        model: Union[SupernetMixin, MeasurableModuleMixin],
        metric_to_state_names: Dict[str, str],
    ):
        self.model = model
        self.metric_to_state_names = metric_to_state_names

    @torch.no_grad()
    def __call__(self, engine: Engine):
        device = get_module_device(self.model)
        with self.model.subnet_masked():
            metrics = {
                k: torch.zeros((1), device=device)
                for k in self.metric_to_state_names.keys()
            }
            metrics = self.model.forward_measurements(metrics)
        # save metrics to state
        for k, v in metrics.items():
            engine.state.metrics[self.metric_to_state_names[k]] = v.item()
