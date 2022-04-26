from rich.console import Console
from rich.table import Table, box

from ignite.engine.engine import Engine
from ignite.engine.events import Events

from torchinfo import summary

from src.torchmodules.mixins import ShapeRecorderMixin


class PrintModelSummary:
    def attach(
        self,
        engine: Engine,
        model: ShapeRecorderMixin,
    ):
        def _fn():
            stats = summary(model, input_size=model.input_shapes(), verbose=0)
            console = Console()
            # table of layers
            table = Table(header_style="bold magenta")
            table.add_column("Name", justify="left", no_wrap=True)
            table.add_column("Params", justify="right")
            table.add_column("In Shape", justify="right", style="white")
            table.add_column("Out Shape", justify="right", style="white")
            for layer in stats.summary_list:
                table.add_row(
                    (" " * layer.depth) + str(layer),
                    f"{layer.num_params:,}",
                    str(layer.input_size),
                    str(layer.output_size),
                )
            console.print(table)
            # totals grid
            grid = Table(expand=False, highlight=True)
            grid.add_column()
            grid.add_column()
            grid.add_row(f"[bold]Total params[/]:", f"{stats.total_params:,}")
            grid.add_row(f"[bold]Trainable params[/]:", f"{stats.total_params:,}")
            grid.add_row(
                f"[bold]Non-trainable params[/]:",
                f"{stats.total_params - stats.trainable_params:,}",
            )
            x = stats.to_readable(stats.total_mult_adds)
            grid.add_row(f"[bold]Total mult-adds ({x[0]})[/]:", "{:0.2f}".format(x[1]))
            grid.add_row(
                f"[bold]Input size (MB)[/]:", f"{stats.to_megabytes(stats.total_input)}"
            )
            grid.add_row(
                f"[bold]Forward/backward pass size (MB)[/]:",
                f"{stats.to_megabytes(stats.total_output_bytes)}",
            )
            grid.add_row(
                f"[bold]Params size (MB)[/]:",
                f"{stats.to_megabytes(stats.total_param_bytes)}",
            )
            grid.add_row(
                f"[bold]Estimated Total Size (MB)[/]:",
                f"{stats.to_megabytes(stats.total_input+stats.total_output_bytes+stats.total_param_bytes)}",
            )
            console.print(grid)
            self.remove.remove()

        self.remove = engine.add_event_handler(Events.ITERATION_COMPLETED(once=1), _fn)

    def __call__(self, *args, **kwds):
        raise ValueError("Use .attach(engine, model, kind) instead.")
