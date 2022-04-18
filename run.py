import dotenv
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from src.types.config import RunConfig
from src.utils.utils import argv_cache
import torch
import gc
import logging
import faulthandler
from kink import di


# Catch SIGSEV error stack trace
# See: https://stackoverflow.com/questions/16731115/how-to-debug-a-python-segmentation-fault
faulthandler.enable()

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs", config_name="train_config.yaml")
def main(config: RunConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils

    # save config in dependency injector
    di["config"] = config

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    utils.extras(config)

    # Pretty print config using Rich library
    if config.print_config:
        utils.print_config(config, resolve=True)

    # Train model
    try:
        score = train(config)
    except Exception as e:
        logging.exception(e)
        raise

    # Clear memory and free torch stuff
    gc.collect()
    torch.cuda.empty_cache()

    return score


def _register_hacks():
    # persistent 'now' resolver
    @argv_cache(
        key_fn=lambda x: f'__rootnow_cache_{x.replace("%", "P").replace("-", "_")}__',
    )
    def _root_now(pattern: str):
        return datetime.now().strftime(pattern)

    OmegaConf.register_new_resolver(
        "root_now",
        _root_now,
        use_cache=True,
        replace=True,
    )

    # dependency kink injector resolver
    OmegaConf.register_new_resolver(
        "di",
        lambda pat: OmegaConf.create({
            "_target_": "kink.di.__getitem__",
            "_args_": [pat],
        }),
        use_cache=False,
        replace=True,
    )

    # eval resolver
    OmegaConf.register_new_resolver(
        "eval",
        lambda x: eval(x),
        use_cache=False,
        replace=True,
    )


if __name__ == "__main__":
    # hydra hacks applied before hydra
    _register_hacks()
    main()
