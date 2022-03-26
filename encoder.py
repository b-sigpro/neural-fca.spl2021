import torch  # noqa
from torch import nn
from torch.nn import functional as fn
from torch.distributions import Normal

from utils.nn import ResSequential


class DepSepConv1d(ResSequential):
    def __init__(self, io_ch, mid_ch, ksize, dilation, dropout):
        super().__init__(
            nn.Conv1d(io_ch, mid_ch, 1),
            nn.PReLU(mid_ch),
            nn.GroupNorm(mid_ch, mid_ch),
            #
            nn.Conv1d(mid_ch, mid_ch, ksize, padding=(ksize - 1) // 2 * dilation, dilation=dilation, groups=mid_ch),
            nn.PReLU(mid_ch),
            nn.GroupNorm(mid_ch, mid_ch),
            #
            nn.Dropout(dropout),
            nn.Conv1d(mid_ch, io_ch, 1))


class DilConvBlock1d(nn.Sequential):
    def __init__(self, io_ch, mid_ch, ksize, n_layers, dropout):
        super().__init__(*[DepSepConv1d(io_ch, mid_ch, ksize, 2 ** ll, dropout) for ll in range(n_layers)])


class Encoder(nn.Module):
    def __init__(self, F=257, K=3, D=50, M=4, N1=256, N2=512, L1=4, L2=8, ksize=3, dropout=0.0, **kwargs):
        super().__init__()

        self.K, self.D = K, D

        self.cnv1 = nn.Sequential(nn.Conv1d((2 * M - 1) * F, N1, 1))
        self.dilcnvs = nn.Sequential(*[DilConvBlock1d(N1, N2, ksize, L2, dropout) for ll in range(L1)])
        self.cnv2 = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(N1, 2*K*D, 1))

        self.eps = 1e-8

    def forward(self, x, distribution=False):
        """
        Parameters
        -------
        x: torch.Tensor([B, T, F, M])
            Multichannel mixture observation
        distribution: bool
            If true, torch.distributions.Normal is returned. Default is False.

        Returns
        -------
        Estimated embedding of each source as torch.Tensor([B, D, T, K]).
        If distribution is True, corresponding Normal distribution is returned
        """

        B, T, F, M = x.shape

        pwrx = torch.abs(x[..., 0]) ** 2
        logx = torch.log(pwrx + self.eps)

        ph  = x[..., 1:] / (x[..., 0, None] + 1e-6)  # [B, T, F, M-1]
        ph /= torch.abs(ph) + 1e-6
        ph  = ph.reshape(B, T, -1)

        feat = torch.cat([logx, ph.real, ph.imag], axis=-1)  # [B, T, M * F]

        h = feat.transpose(1, 2)  # [B, F, T]

        h = self.cnv1(h)
        h = self.dilcnvs(h)

        h = self.cnv2(h).reshape(B, self.K, 2, self.D, T).permute(2, 0, 3, 4, 1)  # [B, 2, D, T, K]

        if distribution:
            return Normal(h[0], fn.softplus(h[1]) + 1e-6)
        else:
            return h[0]
