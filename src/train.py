import logging
from typing import Optional
from src.datamodules.interface import DatamoduleInterface
from src.types.config import (
    IgniteHandler,
    IgniteLogger,
    IgniteLoggerHandler,
    TrainConfig,
)
import hydra
from kink import di
from ignite.engine.engine import Engine
import torch.nn as nn

from src.utils.utils import seed_everything

log = logging.getLogger(__name__)


def train(config: TrainConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all objects from config and runs train/test loops.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.seed is not None:
        seed_everything(config.seed)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DatamoduleInterface = hydra.utils.instantiate(config.datamodule)
    di["datamodule"] = datamodule
    log.info("Running datamodule setup")
    datamodule.setup_data()
    datamodule.setup_splits()

    # Instantiate model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: nn.Module = hydra.utils.instantiate(config.model)
    if config.device is not None:
        model = model.to(device=config.device)
    di["model"] = model
    di["model_params"] = model.parameters()

    # Instantiate trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Engine = hydra.utils.instantiate(config.trainer)
    di["trainer"] = trainer

    # Instantiate handlers
    for name, hd in config.handlers.items():
        log.info(f"Instantiating handler <{name}>")
        handler = hydra.utils.instantiate(hd.handler)
        event = hydra.utils.instantiate(hd.event) if 'event' in hd else None
        kwargs = hd.get("kwargs", {})
        if hasattr(handler, "attach"):
            handler.attach(trainer, **kwargs)
        else:
            trainer.add_event_handler(event, handler, **kwargs)

    # Instantiate loggers
    loggers = []
    for name, lg in config.loggers.items():
        log.info(f"Instantiating logger <{name}>")
        logger = hydra.utils.instantiate(lg.logger)
        loggers.append(logger)
        di[f"loggers.{name}"] = logger
        for hd_name, hdl in lg.handlers.items():
            log.info(f"Attaching logger handler <{hd_name}>")
            handler = hydra.utils.instantiate(hdl.log_handler)
            event = hydra.utils.instantiate(hdl.event) if 'event' in hdl else None
            logger.attach(
                trainer,
                event_name=event,
                log_handler=handler,
            )

    # Run trainer
    log.info(f"Running trainer <{config.trainer._target_}>")
    trainer.run(
        datamodule.train_dataloader(),
        max_epochs=config.max_epochs,
    )

    # close all loggers
    for l in loggers:
        l.close()
