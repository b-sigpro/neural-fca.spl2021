import torch


def to_complex(r):
    return torch.complex(r, torch.zeros_like(r))
