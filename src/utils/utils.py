import logging
import os
import warnings
from typing import Callable, Any, Optional
import numpy as np
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pynvml.smi import nvidia_smi
import backoff
import random
import portalocker
import torch
import tensorflow as tf
from boltons.strutils import bytes2human
from slugify import slugify
from functools import wraps
import sys
import gc
from pathlib import Path

from src.types.config import RunConfig, RunMode


def seed_everything(seed: int):
    log = logging.getLogger(__name__)
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)


def get_module_device(module: torch.nn.Module) -> torch.device:
    """
    Retruns the given module's current device.
    """
    return next(module.parameters()).device


def argv_cache(
    *,
    key_fn,
    serializer=str,
    deserializer=lambda x: x,
) -> Callable:
    def decorator(fn):
        @wraps(fn)
        def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
            key = key_fn(*args, **kwargs)
            key = "__argv_cache__." + key
            log = logging.getLogger(__name__)
            val = None
            for v in sys.argv:
                if v.startswith(f"++{key}="):
                    val = v[3 + len(key) :]
                    log.debug(f"Loaded argv cached value '{key}={val}'")
                    val = deserializer(val)
                    break
            if val is None:
                val = fn(*args, **kwargs)
                srval = serializer(val)
                sys.argv.append(f"++{key}={srval}")
                log.debug(f"Stored argv cache value '{key}'={srval}")
            return val

        return wrapped_fn

    return decorator


@argv_cache(
    key_fn=lambda *a, **k: f"f_{slugify(Path.cwd().name, separator='_')}.auto_select_gpus",
    serializer=lambda l: "-".join(map(str, l)),
    deserializer=lambda s: [int(x) for x in s.split("-")],
)
def auto_select_gpus(
    n: int,
    freemem_ratio_above: float = 0.99,
    fill_mem_upto: float = 0.50,
    fetch_exact_n: bool = True,
    backoff_waitgen=None,
    max_time=None,
    max_tries=None,
    as_str: bool = True,
):
    """
    Retruns a list of cuda device idx which have used memory % below given threshold.
    """
    log = logging.getLogger(__name__)
    log.info(
        f"Performing custom auto select GPUs: n={n}, freemem_ratio_above={freemem_ratio_above}, fill_mem_upto={fill_mem_upto}, fetch_exact_n={fetch_exact_n}"
    )
    nvsmi = nvidia_smi.getInstance()

    if backoff_waitgen is None:

        def _gen():
            while True:
                yield random.uniform(0, 600)

        backoff_waitgen = _gen

    @backoff.on_predicate(
        backoff_waitgen,
        predicate=lambda x: ((fetch_exact_n is True) and (len(x) != n)),
        logger=log,
        max_time=max_time,
        max_tries=max_tries,
    )
    def _fetch_gpus():
        gpus = nvsmi.DeviceQuery(
            "index, memory.total, memory.used, memory.free, utilization.gpu"
        )["gpu"]

        def _proc_mem(p):
            mult = {
                "B": 1,
                "KiB": 1024,
                "MiB": 1024**2,
                "GiB": 1024**3,
            }[p["unit"]]
            return {k: int(v * mult) for k, v in p.items() if k != "unit"}

        def _try_or(fn, default):
            try:
                return fn()
            except Exception:
                return default

        gpus = [
            (
                _try_or(lambda: int(x["minor_number"]), 0),
                _proc_mem(x["fb_memory_usage"]),
                x["utilization"]["gpu_util"],
            )
            for x in gpus
        ]
        # Filter eligible
        rt = []
        for i, m, util in reversed(gpus):
            m_req = int(m["total"] * freemem_ratio_above)
            m_avail = int(m["free"] + torch.cuda.memory_reserved(i))
            m_fill = int(fill_mem_upto * m["total"])
            log.info(
                f"GPU {i} [Util: {util}%] [Avail: {bytes2human(m_avail)}({m_avail}B)] [Req: {bytes2human(m_req)}({m_req}B)] [Fill: {bytes2human(m_fill)}({m_fill}B)]"
            )
            if m_avail >= m_req:
                rt.append(
                    (
                        i,
                        m,
                        util,
                    )
                )
        # select best n with least util
        rt = sorted(rt, key=lambda t: t[-1])[:n]
        return rt

    # ensure inter-process serialisation across jobs
    with portalocker.Lock("/tmp/auto_select_gpus.lock", timeout=1e32):
        rt = _fetch_gpus()
        log.info(f"Selected GPUs: {rt}")
        # fill selected GPU's memory to prevent others from selecting same GPU
        # See: https://gist.github.com/sparkydogX/845b658e3e6cef58a7bf706a9f43d7bf
        for i, m, util in rt:
            # m.free is allocatable memory in bytes
            # torch.float32 is 4 bytes
            to_fill = ((m["total"] * fill_mem_upto) - m["used"]) / 4
            log.info(f"Filling {bytes2human(to_fill*4)} memory for selected GPU {i}.")
            # to_fill = math.ceil(m.total * 0.95 / 4)
            # allocate in chunks to counter fragmentation
            n_chunk_levels = [1, 2, 10, 100, 1e3]
            trash = []
            for sz in n_chunk_levels:
                cnt = 0
                sh = (int(to_fill / sz),)
                for _ in range(sh[0]):
                    try:
                        x = torch.empty(
                            sh,
                            dtype=torch.float32,
                            device=f"cuda:{i}",
                        )
                        trash.append(x)
                        to_fill -= sh[0]
                        cnt += 1
                    except Exception:
                        log.debug(
                            f"Did {cnt} allocations of shape {sh} at chunking level {sz}. ({bytes2human(cnt*sh[0]*4)})"
                        )
                        break
            # free memory
            del trash
            for _ in range(10):
                gc.collect()
            # log memory seen by torch
            log.debug(
                f"Post fill memory: Allocated({bytes2human(torch.cuda.memory_allocated(i))}), Reserved({bytes2human(torch.cuda.memory_reserved(i))})"
            )
        rt, _, _ = zip(*rt)
        rt = list(rt)
        if as_str:
            rt = [f'cuda:{i}' for i in rt]
    return rt


def extras(config: RunConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = logging.getLogger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.ignore_warnings:
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.run_mode == RunMode.experiment and config.name is None:
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # Set log level for datasets logger
    logging.getLogger("datasets.builder").setLevel(logging.ERROR)


def print_config(
    config: DictConfig,
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def log_hyperparameters(
    config: DictConfig,
    model: torch.nn.Module,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionally saves:
        - number of model parameters
    """

    # Log everything
    hparams = OmegaConf.to_container(config, resolve=True)

    # choose which parts of hydra config will be saved to loggers
    # hparams["trainer"] = config["trainer"]
    # hparams["model"] = config["model"]
    # hparams["datamodule"] = config["datamodule"]
    if hasattr(model, "model_name"):
        hparams["model_name"] = model.model_name

    # if "seed" in config:
    #     hparams["seed"] = config["seed"]
    # if "callbacks" in config:
    #     hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    # TODO do something to log hparams
    # trainer.logger.log_hyperparams(hparams)


# def finish(
#     config: DictConfig,
#     model: pl.LightningModule,
#     datamodule: pl.LightningDataModule,
#     trainer: pl.Trainer,
#     callbacks: List[pl.Callback],
#     logger: List[pl.loggers.LightningLoggerBase],
# ) -> None:
#     """Makes sure everything closed properly."""
#     # TODO make logging end properly
#     # without this, sweeps with wandb logger might crash!
#     for lg in logger:
#         if isinstance(lg, pl.loggers.wandb.WandbLogger):
#             import wandb

#             # upload log and config files
#             exp: wandb.wandb_run.Run = lg.experiment
#             exp.save(".hydra/*")
#             exp.save("*.log")
#             # finish wandb
#             wandb.finish()
