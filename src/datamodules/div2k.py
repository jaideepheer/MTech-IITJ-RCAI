from typing import Dict, Literal, Tuple

import torch
from src.datamodules.interface import DatamoduleInterface
import tensorflow_datasets as tfds
import tensorflow as tf

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

        def _image_norm_transform(image: tf.Tensor):
            image = tf.cast(image, dtype=tf.float32)
            image /= 255.0
            # rescale to [-1,1]
            image = 2 * image - 1
            return image

        def _map_fn(dt):
            hr = dt["hr"]
            lr = dt["lr"]
            # do upscaling
            if self.lr_upscaling is not None:
                lr = tf.image.resize(lr, tf.shape(hr)[-3:-1], method=self.lr_upscaling)
            # normalize images
            lr = _image_norm_transform(lr)
            hr = _image_norm_transform(hr)
            # change to nchw format
            lr = tf.transpose(lr, [2, 0, 1])
            hr = tf.transpose(hr, [2, 0, 1])
            return {
                "hr": hr,
                "lr": lr,
            }

        def _post_cache_map(dt):
            seed = tf.random.get_global_generator().make_seeds(count=2)
            for k in ["lr", "hr"]:
                # random crop
                dt[k] = tf.image.stateless_random_crop(
                    dt[k],
                    size=(3, self.patch_size, self.patch_size),
                    seed=seed[:, 0],
                )
                # random horizontal flip
                dt[k] = tf.image.stateless_random_flip_left_right(
                    dt[k],
                    seed=seed[:, 1],
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
