from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import torch.nn as nn
from src.datamodules.interface import DatamoduleInterface


class RunMode(Enum):
    default = "default"
    experiment = "experiment"
    debug = "debug"


@dataclass
class IgniteLoggerHandler:
    event: Any
    log_handler: Any
    # names of engines to attach to, if not provided, attaches to every engine
    engines: Optional[List[str]]


@dataclass
class IgniteLogger:
    logger: Any
    handlers: Optional[Dict[str, IgniteLoggerHandler]]


@dataclass
class RunConfig:
    # path to original working directory
    # https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
    work_dir: str
    # path to dataset directory
    data_dir: str
    # the mode of the run to decide directories and other settings
    run_mode: RunMode
    # disable python warnings if they annoy you
    ignore_warnings: bool
    # seed for random number generators in pytorch, tensorflow, numpy and python.random
    seed: Optional[int]
    # name of the run is accessed by loggers
    # should be used along with experiment mode
    name: Optional[str]
    # print config
    print_config: bool
    # dict of loggers to use
    loggers: Optional[Dict[str, IgniteLogger]]


@dataclass
class IgniteHandler:
    handler: Callable
    event: Optional[Any]
    kwargs: Optional[Any]
    args: Optional[List[Any]]


@dataclass
class IgniteEngine:
    # ignite engine instance
    engine: Any
    # event handlers for this engine
    handlers: Optional[Dict[str, Any]]
    # to run
    run_engine: bool


@dataclass
class TrainConfig(RunConfig):
    # the device to train on
    device: Optional[Any]
    # model
    model: Any
    # datamodule
    datamodule: Any
    # engines
    engines: Dict[str, IgniteEngine]


cs = ConfigStore.instance()
cs.store("default_run_config", node=RunConfig)
cs.store("default_train_config", node=TrainConfig)
