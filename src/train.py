import logging
from typing import Optional

from hjson import OrderedDict
from omegaconf import OmegaConf
from src.datamodules.interface import DatamoduleInterface
from src.types.config import (
    IgniteEngine,
    TrainConfig,
)
import hydra
from kink import di
from ignite.engine.engine import Engine
import torch.nn as nn
import tensorflow as tf

from src.utils.utils import seed_everything

log = logging.getLogger(__name__)


def try_instantiate(config):
    try:
        return hydra.utils.instantiate(config)
    except Exception as e:
        log.warn(f"Hydra could not instantiate config: {e}")
        return config


def train(config: TrainConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all objects from config and runs train/test loops.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # disable tf GPU usage
    # See: https://datascience.stackexchange.com/a/76039/134896
    tf.config.set_visible_devices([], 'GPU')

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.seed is not None:
        seed_everything(config.seed)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DatamoduleInterface = try_instantiate(config.datamodule)
    di["datamodule"] = datamodule
    log.info("Running datamodule setup")
    datamodule.setup_data()
    datamodule.setup_splits()

    # Instantiate model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: nn.Module = try_instantiate(config.model)
    if config.device is not None:
        log.info(f"Instantiating device <{config.device}>")
        device = try_instantiate(config.device)
        log.info(f"Selected device: {device}")
        model = model.to(device=device)
    di["model"] = model

    # Instantiate engines
    engines: OrderedDict[str, IgniteEngine] = OrderedDict()
    for name, eg in config.engines.items():
        log.info(f"Instantiating engine <{name}>")
        eg = OmegaConf.to_object(eg)
        eg.engine = try_instantiate(eg.engine)
        engines[name] = eg
        di[f"engines.{name}"] = eg.engine

    # Instantiate loggers
    loggers = []
    for name, lg in config.loggers.items():
        log.info(f"Instantiating logger <{name}>")
        logger = try_instantiate(lg.logger)
        loggers.append(logger)
        di[f"loggers.{name}"] = logger

    # Instantiate engine handlers
    for name, eg in engines.items():
        for hdl_name, hd in eg.handlers.items():
            log.info(f"Instantiating handler <{name}.{hdl_name}>")
            handler = try_instantiate(hd["handler"])
            event = try_instantiate(hd["event"]) if "event" in hd else None
            # support list of events
            if isinstance(event, (list, tuple)):
                e = try_instantiate(event[0])
                for v in event[1:]:
                    e |= try_instantiate(v)
                event = e
            kwargs = try_instantiate(hd["kwargs"]) if "kwargs" in hd else {}
            if hasattr(handler, "attach"):
                handler.attach(eg.engine, **kwargs)
            else:
                eg.engine.add_event_handler(event, handler, **kwargs)

    # Instantiate logger handlers
    for name, lg in config.loggers.items():
        for hd_name, hdl in lg.handlers.items():
            log.info(f"Attaching logger handler <{hd_name}>")
            handler = try_instantiate(hdl.log_handler)
            event = try_instantiate(hdl.event) if "event" in hdl else None
            to_attach = hdl.engines if "engines" in hdl else engines.keys()
            for eg_name in to_attach:
                logger.attach(
                    engines[eg_name].engine,
                    event_name=event,
                    log_handler=handler,
                )

    # Run engines
    for name, engine in engines.items():
        if engine.run_engine:
            log.info(f"Running engine <{name}>")
            engine.engine.run(
                datamodule.train_dataloader(),
                max_epochs=engine.max_epochs,
            )

    # close all loggers
    for l in loggers:
        l.close()
