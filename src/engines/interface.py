from typing import Any, Callable, Iterable, Optional
from ignite.engine.engine import Engine
from ignite.engine.events import State


class EngineInterface(Engine):
    def __init__(
        self,
        process_function: Callable[["Engine", Any], Any],
        **run_args,
    ):
        super().__init__(process_function)
        self.run_args = run_args

    def run(
        self,
        data: Optional[Iterable] = None,
        max_epochs: Optional[int] = None,
        epoch_length: Optional[int] = None,
    ) -> State:
        return super().run(
            data=self.run_args.get("data", data),
            max_epochs=self.run_args.get("max_epochs", max_epochs),
            epoch_length=self.run_args.get("epoch_length", epoch_length),
        )
