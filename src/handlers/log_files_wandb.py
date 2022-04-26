from typing import Dict, Union
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.contrib.handlers.wandb_logger import WandBLogger

class LogFilesToWandb:
    def __init__(self, files_with_policy: Dict[str, str]):
        self.files_with_policy = files_with_policy
    
    def __call__(self, engine: Engine, logger: WandBLogger, event_name: Union[str, Events]):
        for f, policy in self.files_with_policy.items():
            logger._wandb.save(f, policy=policy)