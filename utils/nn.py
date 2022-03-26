import torch  # noqa
from torch import nn


class ResSequential(nn.Sequential):
    def forward(self, x):
        return x + super().forward(x)
