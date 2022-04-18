import torch.nn as nn
import torch
from torch.nn.functional import gumbel_softmax, softmax


class GumbelSoftmax(nn.Module):
    def __init__(self, tau: float, hard: bool = False, dim: int = 0):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim

    def forward(self, x):
        return gumbel_softmax(
            x,
            tau=self.tau,
            hard=self.hard,
            dim=self.dim,
        )

    @property
    def temperature(self):
        return self.tau

    @temperature.setter
    def temperature(self, value):
        self.tau = value


class SoftmaxTemperature(nn.Module):
    def __init__(self, temp: float = 1.0, dim: int = 0):
        super().__init__()
        self.temp = temp
        self.dim = dim

    def forward(self, x):
        x = x / self.temp
        return softmax(x, dim=self.dim)

    @property
    def temperature(self):
        return self.temp

    @temperature.setter
    def temperature(self, value):
        self.temp = value
