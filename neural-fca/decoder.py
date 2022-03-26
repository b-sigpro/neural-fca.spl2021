import torch  # noqa
from torch import nn
from torch.nn import functional as fn

from utils.nn import ResSequential


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()

        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(num_channels))
        self.bias = nn.Parameter(torch.Tensor(num_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        mu = torch.mean(x, axis=(1, 2), keepdims=True)
        sig = torch.sqrt(torch.mean((x - mu) ** 2, axis=(1, 2), keepdims=True) + self.eps)

        return (x - mu) / sig * self.weight[:, None, None] + self.bias[:, None, None]


class ResConvBlock2d(ResSequential):
    def __init__(self, io_ch, dropout):
        super().__init__(nn.Dropout(dropout), nn.Conv2d(io_ch, io_ch, 1), nn.PReLU(io_ch), LayerNorm(io_ch))


class Decoder(nn.Module):
    def __init__(self, F=257, K=3, D=50, L=3, N=256, dropout=0.0, **kwargs):
        super().__init__()

        self.cnv = nn.Sequential(nn.Conv2d(D, N, 1), *[ResConvBlock2d(N, dropout) for ll in range(L-1)], nn.Conv2d(N, F, 1))

    def forward(self, z):
        """
        Parameters
        -------
        z: torch.Tensor([B, D, T, K])
            Multichannel mixture observation

        Returns
        -------
        Estimated power spectral density of each source as torch.Tensor([B, T, F, K]).
        """
        h = self.cnv(z).permute(0, 2, 1, 3)

        lm = fn.softplus(h) + 1e-6

        return lm
