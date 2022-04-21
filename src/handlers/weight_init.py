import logging
import torch
import torch.nn as nn
from enum import Enum
from ignite.engine.engine import Engine
from ignite.engine.events import Events

log = logging.getLogger(__name__)


class ModelWeightInitKind(str, Enum):
    ORTHOGONAL = "orthogonal"


class ModelWeightHandler:
    @torch.no_grad()
    def _orthogonal_init(self, m: nn.Module):
        if hasattr(m, 'weight'):
            cls_name = m.__class__.__name__
            if "Conv" in cls_name:
                nn.init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif "Linear" in cls_name:
                nn.init.orthogonal_(m.weight.data, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif "BatchNorm2d" in cls_name:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    def attach(self, engine: Engine, model: nn.Module, kind: ModelWeightInitKind):
        fn = None
        if kind == ModelWeightInitKind.ORTHOGONAL:
            fn = lambda: model.apply(self._orthogonal_init)
        else:
            raise NotImplementedError(f"Invalid model weight init kind: {kind}")

        def _fn():
            log.info(f"Initializing model weights as {kind}.")
            fn()
            self.remove.remove()

        self.remove = engine.add_event_handler(Events.ITERATION_COMPLETED(once=1), _fn)

    def __call__(self, *args, **kwds):
        raise ValueError("Use .attach(engine, model, kind) instead.")
