from typing import Dict, Literal, Tuple

import torch
from src.datamodules.interface import DatamoduleInterface
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random

Div2kSubset_T = Literal[
    "bicubic_x2",
    "bicubic_x3",
    "bicubic_x4",
    "bicubic_x8",
    "realistic_difficult_x4",
    "realistic_mild_x4",
    # This contains 4 samples of the same image for some reason
    "realistic_wild_x4",
    "unknown_x2",
    "unknown_x3",
    "unknown_x4",
    # this is only valid when lr_interpolation != 'none'
    # "mixed",
]


class Div2kDatamodule(DatamoduleInterface):
    def __init__(
        self,
        *_,
        data_dir: str = "data",
        batch_size: int,
        subset: Div2kSubset_T,
        patch_size: int,
        # this is sent to tfds.load and can use tfds split format, eg. train[80%]
        train_split: str = "train[:-100]",
        test_split: str = "train[-100:]",
        validation_split: str = "validation",
        lr_upscaling: tf.image.ResizeMethod = tf.image.ResizeMethod.BICUBIC,
        in_memory: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.subset = subset
        self.splits = {
            "train": train_split,
            "test": test_split,
            "validation": validation_split,
        }
        self.lr_upscaling = lr_upscaling
        self.in_memory = in_memory

    @property
    def data_shapes(self) -> Tuple[torch.Size]:
        """
        Returns a tuple containing shapes of the data tensors.
        """
        raise NotImplementedError()

    def setup_data(self):
        """
        Called to download/extract data.
        """
        self.datasets: Dict[str, tf.data.Dataset] = {}
        self.seed = random.random()

        def _map_fn(dt):
            hr = dt["hr"]
            lr = dt["lr"]
            # normalize images
            hr = tf.image.per_image_standardization(hr)
            lr = tf.image.per_image_standardization(lr)
            if self.lr_upscaling is not None:
                lr = tf.image.resize(lr, tf.shape(hr)[-3:-1], method=self.lr_upscaling)
            # change to nchw format
            lr = tf.transpose(lr, [0, 3, 1, 2])
            hr = tf.transpose(hr, [0, 3, 1, 2])
            return {
                "hr": hr,
                "lr": lr,
            }

        def _post_cache_map(dt):
            dt = {
                k: tf.image.random_crop(
                    v,
                    size=(self.batch_size, 3, self.patch_size, self.patch_size),
                    seed=self.seed,
                )
                for k, v in dt.items()
            }
            return dt

        for mode in ["train", "test", "validation"]:
            ds = tfds.load(
                f"div2k/{self.subset}",
                split=self.splits[mode],
                batch_size=self.batch_size,
                data_dir=self.data_dir,
            ).map(
                _map_fn,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            if self.in_memory:
                ds = ds.cache()
            # random crop and mutations
            ds = ds.map(
                _post_cache_map,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            # prefetch
            ds = ds.prefetch(tf.data.AUTOTUNE)
            # auto prefetch next batch
            self.datasets[mode] = tfds.as_numpy(ds)

    def setup_splits(self):
        """
        Called to generate data splits.
        """
        ...

    def train_dataloader(self):
        """
        Returns training dataloader.
        """
        return self.datasets["train"]

    def validation_dataloader(self):
        """
        Returns validation dataloader.
        """
        return self.datasets["validation"]

    def test_dataloader(self):
        """
        Returns test dataloader.
        """
        return self.datasets["test"]
