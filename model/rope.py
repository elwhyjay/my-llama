import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Precompute the frequencies for rotary embeddings.

    Args:
        dim (int): Dimension of the model.
        end (int): Maximum sequence length.
        theta (float, optional): Base frequency. Defaults to 10000.0.
        device (Optional[torch.device], optional): Device to place the tensor on. Defaults to None.

    Returns:
        torch.Tensor: Precomputed frequencies tensor of shape (end, dim // 2, 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim//2].float() / dim))
    t = torch.arange(end, device = freqs.device).float()
    freqs = torch.einsum("i,j->ij", t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcasting(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape the precomputed frequencies for broadcasting.

    Args:
        freqs_cis (torch.Tensor): Precomputed frequencies tensor of shape (seq_len, dim // 2, 2).
        x (torch.Tensor): Input tensor of shape (..., seq_len, dim).

    Returns:
        torch.Tensor: Reshaped frequencies tensor for broadcasting.
    """
    x_dim = x.ndim
    assert 1 < x_dim , "Input tensor must have at least 2 dimensions"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) , "Frequency tensor shape mismatch"
    shape = [
        d if i == 1 or i == x_dim - 1 else 1
        for i, d in enumerate(x.shape)
    ]   
    freqs_cis = freqs_cis.view(*shape)
    return freqs_cis

def rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor of shape (..., seq_len, dim).
        xk (torch.Tensor): Key tensor of shape (..., seq_len, dim).
        freqs_cis (torch.Tensor): Precomputed frequencies tensor of shape (seq_len, dim // 2, 2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors after applying rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcasting(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)