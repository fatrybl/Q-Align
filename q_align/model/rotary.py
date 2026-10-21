"""The rotary table of transformers 4.36's Llama, which the released Q-Align runs with::

    θ_i = base^(−2i/d),   φ(t, i) = t·θ_i,   cos φ, sin φ   over [φ(t, ·), φ(t, ·)],

all in float32 on the CPU, the device 4.36 built it on when the model was created. Version 5
computes its own table on the activations' device, where float32 cosines may round apart.
"""

import torch
from torch import Tensor

# The device 4.36 built the table on, the default one when the model was created.
_DEVICE = torch.device("cpu")


def rotary_table(dim: int, positions: int, base: float) -> tuple[Tensor, Tensor]:
    """Cosines and sines of the rotary angles, ``[positions, dim]`` each.

    Args:
        dim: Size of an attention head.
        positions: Positions in the table, the model's context length.
        base: The rotary base.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=_DEVICE).float() / dim))
    t = torch.arange(positions, device=_DEVICE, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    table = (emb.cos(), emb.sin())
    return table
