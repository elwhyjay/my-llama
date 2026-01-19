import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, scale_to_hidden_dim: int, ffn_dim_multiplier: Optional[float] = None):
        """
        SwiGLU implemnt. 
        SwiGLU (x) = SiLU(W1*x) * (W2*x)
        Args:
            dim_in (int): Input dimension.
            dim_out (int): Output dimension.
            scale_to_hidden_dim (int): Scale to hidden dimension.
            ffn_dim_multiplier (Optional[float], optional): Multiplier for feedforward dimension. Defaults to None.        
        """
        super().__init__()
        dim_out = int(2*dim_out/3)
        if ffn_dim_multiplier is not None:
            dim_out = int(dim_out * ffn_dim_multiplier)
        dim_out = scale_to_hidden_dim*((dim_out + scale_to_hidden_dim - 1)//scale_to_hidden_dim)

        self.w1 = nn.Linear(dim_in, dim_out,bias =False)
        
        self.w2 = nn.Linear(dim_in, dim_out,bias =False)
        self.w3 = nn.Linear(dim_out, dim_in,bias =False)
        self.activation = SiLU()
    def forward(self, x):
        return self.w3(self.activation(self.w1(x)) * self.w2(x))

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim : int , scale_to_hidden_dim : int, ffn_dim_multiplier:Optional[float], dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.ffn = SwiGLU(dim, hidden_dim,scale_to_hidden_dim,ffn_dim_multiplier)
    def forward(self, x):
        return self.ffn(x)
        