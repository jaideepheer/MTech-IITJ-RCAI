from typing import Dict, Literal, Tuple

import torch
from src.datamodules.interface import DatamoduleInterface
import tensorflow_datasets as tfds
import tensorflow as tf
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
        batch_size_train: int,
        batch_size_validation: int,
        batch_size_test: int,
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
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.batch_size_validation = batch_size_validation
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

        def _per_image_norm_tensor(image: tf.Tensor):
            image = tf.cast(image, dtype=tf.float32)
            num_pixels = tf.reduce_prod(tf.shape(image)[-3:])
            image_mean = tf.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)
            # Apply a minimum normalization that protects us against uniform images.
            stddev = tf.math.reduce_std(image, axis=[-1, -2, -3], keepdims=True)
            min_stddev = tf.math.rsqrt(tf.cast(num_pixels, tf.float32))
            adjusted_stddev = tf.maximum(stddev, min_stddev)
            # apply normalization
            image -= image_mean
            image = tf.divide(image, adjusted_stddev)
            return {
                "image": image,
                # / 255.0 is done to ensure de-normalisation keeps values in [0,1]
                "mean": image_mean / 255.0,
                "stddev": stddev / 255.0,
            }

        def _map_fn(dt):
            hr = dt["hr"]
            lr = dt["lr"]
            # do upscaling
            if self.lr_upscaling is not None:
                lr = tf.image.resize(lr, tf.shape(hr)[-3:-1], method=self.lr_upscaling)
            # normalize images
            lr_meta = _per_image_norm_tensor(lr)
            hr_meta = _per_image_norm_tensor(hr)
            lr = lr_meta["image"]
            hr = hr_meta["image"]
            # change to nchw format
            lr = tf.transpose(lr, [2, 0, 1])
            hr = tf.transpose(hr, [2, 0, 1])
            return {
                "hr": {
                    **hr_meta,
                    "image": hr,
                },
                "lr": {
                    **lr_meta,
                    "image": lr,
                },
            }

        def _post_cache_map(dt):
            seed = tf.random.get_global_generator().make_seeds(count=1)[:, 0]
            for k in ["lr", "hr"]:
                dt[k]["image"] = tf.image.stateless_random_crop(
                    dt[k]["image"],
                    size=(3, self.patch_size, self.patch_size),
                    seed=seed,
                )
            return dt

        for mode in ["train", "test", "validation"]:
            ds = tfds.load(
                f"div2k/{self.subset}",
                split=self.splits[mode],
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
            # save mapped dataset
            self.datasets[mode] = ds

    def setup_splits(self):
        """
        Called to generate data splits.
        """
        ...

    def train_dataloader(self):
        """
        Returns training dataloader.
        """
        return tfds.as_numpy(
            self.datasets["train"]
            .batch(
                self.batch_size_train,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def validation_dataloader(self):
        """
        Returns validation dataloader.
        """
        return tfds.as_numpy(
            self.datasets["validation"]
            .batch(
                self.batch_size_validation,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )

    def test_dataloader(self):
        """
        Returns test dataloader.
        """
        return tfds.as_numpy(
            self.datasets["test"]
            .batch(
                self.batch_size_test,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )
