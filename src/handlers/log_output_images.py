from typing import Dict, List, Literal, Union
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.contrib.handlers.base_logger import BaseLogger
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import torch
import einops
import torchvision.transforms.functional as F


class LogOutputImages:
    def __init__(
        self,
        *_,
        tag: str,
        metric_names: List[str],
        n_images: int,
        orientation: Literal["vertical", "horizontal"] = "horizontal",
    ):
        """
        Takes images from engine output, stitches them in vertical/horizontal order and logs result to logger.

        Parameters
        ----------
        tag: str
            Tag for logging.
        metric_names: List[str]
            Keys in engine output to treat as images.
            Images are stitched in this order.
        n_images
            No. of images to log.
        orientation: Literal['vertical', 'horizontal']
            Orientation to stitch images in.
        """
        self.orientation = orientation
        self.tag = tag
        self.metric_names = metric_names
        self.n_images = n_images
        self._image_cache = []
        self.enabled = False

    def _reset_state(self):
        self._image_cache.clear()
        self.enabled = True

    @torch.no_grad()
    def _log_images(self, images: torch.Tensor):
        logger = self.logger
        if isinstance(logger, TensorboardLogger):
            # merge images to single large image
            images = einops.rearrange(
                images,
                f"n h w c -> {'h (n w)' if self.orientation == 'vertical' else '(n h) w'} c",
            )
            logger.writer.add_image(
                img_tensor=images,
                tag=self.tag,
                dataformats="HWC",
                global_step=self.global_step_engine.state.epoch,
            )

    @torch.no_grad()
    def _on_iter_complete(self):
        if self.enabled:
            # filter output keys and store in cache
            output: Dict[str, torch.Tensor] = self.engine.state.output
            output = [output[k].detach() for k in self.metric_names]
            if len(self._image_cache) == 0:
                self._image_cache = output
            else:
                self._image_cache = [
                    torch.cat([k, v], dim=0) for k, v in zip(self._image_cache, output)
                ]
            # if this iteration satisfies n_images req., log image data to logger
            if self._image_cache[0].shape[0] >= self.n_images:
                # select required images
                images = [i[: self.n_images] for i in self._image_cache]
                # clear cache
                self._image_cache.clear()
                self.enabled = False
                # get largest hxw shape
                hw = [tuple(img.shape[-2:]) for img in images]
                max_height, max_width = [max(s) for s in zip(*hw)]
                # resize images and stack them
                images = torch.stack(
                    [
                        F.resize(
                            i, [max_height, max_width], F.InterpolationMode.NEAREST
                        )
                        for i in images
                    ],
                    dim=1,
                )
                # reshape to orientation
                images = einops.rearrange(
                    images,
                    f"b (i mt) c h w -> b {'(i mt h) w' if self.orientation == 'vertical' else 'h (i mt w)'} c",
                    mt=len(self.metric_names),
                )
                # log images
                self._log_images(images)

    def attach(
        self,
        engine: Engine,
        logger: BaseLogger,
        global_step_engine: Engine = None,
        trigger_event: Union[str, Events] = Events.EPOCH_STARTED,
    ):
        self.engine = engine
        self.logger = logger
        self.global_step_engine = global_step_engine or engine
        engine.add_event_handler(trigger_event, self._reset_state)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self._on_iter_complete)

    def __call__(self, engine: Engine, logger, event: Union[str, Events]):
        raise NotImplementedError("Use .attach(engine, logger) to attach this handler.")
