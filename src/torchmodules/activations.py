import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax, softmax


class GumbelSoftmax(nn.Module):
    def __init__(self, temperature: float, hard: bool = False, dim: int = 0):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
        self.dim = dim

    def forward(self, x):
        return gumbel_softmax(
            x,
            tau=float(self.temperature),
            hard=self.hard,
            dim=self.dim,
        )


class SoftmaxTemperature(nn.Module):
    def __init__(self, temperature: float = 1.0, dim: int = 0):
        super().__init__()
        self.temperature = temperature
        self.dim = dim

    def forward(self, x):
        x = x / self.temperature
        return softmax(x, dim=self.dim)
