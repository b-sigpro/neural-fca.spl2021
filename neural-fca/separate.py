#! /usr/bin/env python3

import pickle as pkl
from argparse import ArgumentParser
from progressbar import ProgressBar

import numpy as np

import torch
from torch import nn

from opt_einsum import contract

import soundfile as sf
from librosa.core import stft, istft

from utils.functional import to_complex

from encoder import Encoder
from decoder import Decoder


def initialize(args):
    device = torch.device("cuda" if args.gpu else "cpu")

    enc = Encoder()
    enc.load_state_dict(torch.load(f"{args.model_path}/encoder.pt", map_location=device))
    enc.eval()

    dec = Decoder()
    dec.load_state_dict(torch.load(f"{args.model_path}/decoder.pt", map_location=device))
    dec.eval()

    return enc, dec, device


@torch.enable_grad()
def finetune(x, z0, dec, lr, n_iter, n_ziter, n_pre_iter, out_ch, eps=1e-8, debug=True, **kargs):
    T, F, M = x.shape
    D, T, N = z0.shape

    device = x.device

    xx = contract("ftm,ftn->ftmn", x, x.conj())

    # initialize z & opt
    z = nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=lr)
    opt.zero_grad()

    # initialize lm & H
    lm = dec(z[None])[0]
    H = torch.tile(torch.eye(M, dtype=torch.complex64, device=device), (F, N, 1, 1))

    eI = 1e-6 * torch.eye(M, dtype=torch.complex64, device=device)

    # main loop
    pbar = ProgressBar(redirect_stdout=True) if debug else (lambda x: x)
    for ii in pbar(range(n_iter)):
        # update H
        with torch.no_grad():
            Yk = contract("tfk,fkmn->tfkmn", to_complex(lm), H)
            Y = Yk.sum(axis=2) + eI  # [T, F, M, M]
            Yi = torch.linalg.inv(Y)

            YixxYi = contract("tfmn,tfno,tfop->tfmp", Yi, xx, Yi)
            Z = Yk + contract("tfkmn,tfno,tfkop->tfkmp", Yk, YixxYi - Yi, Yk)

            H = contract("tfkmn,tfk->fkmn", Z, to_complex(1 / lm)) / T + eI

        # update z
        for jj in range(n_ziter) if ii >= n_pre_iter else []:
            Y = contract("tfk,fkmn->tfmn", to_complex(lm), H) + eI
            Yi = torch.linalg.inv(Y)

            _, ldY = torch.slogdet(Y)
            loss = ldY.sum() + contract("tfmn,tfnm->", xx, Yi).real

            loss.backward()
            opt.step()

            opt.zero_grad()
            lm = dec(z[None])[0]

    # multichannel Wiener filter
    with torch.no_grad():
        Yk = contract("tfk,fkmn->tfkmn", dec(z[None])[0], H)
        Y = Yk.sum(axis=2) + eI  # [T, F, M, M]
        Yi = torch.linalg.inv(Y)

        s = contract("tfkmn,tfno,tfo->ktfm", Yk, Yi, x)[..., out_ch]

    return s, lm, z, H


@torch.no_grad()
def separate(src_fname, dst_fname, enc, dec, device, args):
    # load wav
    wav, sr = sf.read(src_fname, always_2d=True)
    if wav.shape[1] > 4:
        wav = wav[:, :4]

    assert sr == 16000

    # generate spectrogram
    x = np.stack([stft(w, n_fft=512, hop_length=128).T for w in wav.T], axis=-1)
    x = torch.as_tensor(x, dtype=torch.complex64, device=device)
    T, F, M = x.shape

    # normalize
    scale = x.abs().square().mean().sqrt()
    x /= scale

    # estimate latent features z
    z0 = enc(x[None])[0]

    # separate sources to maximize p(x | z, H)
    s, lm, z, H = finetune(x, z0, dec, **vars(args))

    # dump wav
    s *= scale
    dst_wav = np.stack([istft(s_.T.cpu().numpy(), hop_length=128) for s_ in s ], axis=-1)
    sf.write(dst_fname, dst_wav, sr)

    return x[..., 0].cpu().numpy(), lm.cpu().numpy()


if __name__ == '__main__':
    import matplotlib.pyplot as plt  # noqa

    parser = ArgumentParser()
    parser.add_argument('model_path'  , type=str)
    parser.add_argument("src_wav"     , type=str)
    parser.add_argument("dst_wav"     , type=str)
    parser.add_argument("--lr"        , type=float, default=0.2)
    parser.add_argument("--n_iter"    , type=int  , default=200)
    parser.add_argument("--n_ziter"   , type=int  , default=1)
    parser.add_argument("--n_pre_iter", type=int  , default=0)
    parser.add_argument("--out_ch"    , type=int  , default=0)
    parser.add_argument("--gpu"       , action="store_true")
    parser.add_argument("--plot"      , action="store_true")
    args = parser.parse_args()

    # initialize network
    enc, dec, device = initialize(args)

    # separate
    x, lm = separate(args.src_wav, args.dst_wav, enc, dec, device, args)

    if args.plot:
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=[8, 8])

        axs[0].imshow(np.log(np.abs(x)).T, aspect="auto", origin="lower")
        for n, ax in enumerate(axs[1:4]):
            ax.imshow(np.log(lm[..., n]).T, aspect="auto", origin="lower")

        fig.tight_layout(pad=0.1)
        fig.savefig("neural-fca.png")
