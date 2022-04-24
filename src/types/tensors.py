from dataclasses import dataclass
from typing import Dict
import torch
from torchtyping import TensorType


@dataclass
class ImageTensor:
    normalized_image: TensorType["N", "C", "H", "W"]
    original_mean: TensorType["N", "C", "H", "W"]
    original_stddev: TensorType["N", "C", "H", "W"]

    @property
    def device(self):
        return self.normalized_image.device

    @property
    def batch_size(self):
        return self.normalized_image.shape[0]

    def to(self, *a, **k):
        self.normalized_image = self.normalized_image.to(*a, **k)
        self.original_mean = self.original_mean.to(*a, **k)
        self.original_stddev = self.original_stddev.to(*a, **k)
        return self

    def detach(self):
        self.normalized_image = self.normalized_image.detach()
        self.original_mean = self.original_mean.detach()
        self.original_stddev = self.original_stddev.detach()
        return self

    @staticmethod
    def from_dict(data: Dict[str, torch.Tensor], dtype=None, device=None):
        image = torch.as_tensor(data["image"], dtype=dtype, device=device)
        mean = torch.as_tensor(data["mean"], dtype=dtype, device=device)
        stddev = torch.as_tensor(data["stddev"], dtype=dtype, device=device)
        return ImageTensor(image, mean, stddev)

    def de_normalize(self) -> torch.Tensor:
        # de-normalize
        img = torch.multiply(self.normalized_image, self.original_stddev)
        img += self.original_mean
        return img

    def transform_to_stats(self, mean, stddev) -> "ImageTensor":
        """
        Transform image to given stats.
        See: https://stats.stackexchange.com/a/459972
        """
        image = torch.divide(
            self.de_normalize() - self.original_mean,
            self.original_stddev,
        )
        return ImageTensor(image, mean, stddev)
