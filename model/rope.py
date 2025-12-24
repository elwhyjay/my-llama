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