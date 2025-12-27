import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class AttentionConfig:
    hidden_dim: int = 4096
    numb_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None

    vocab_size: int = -1 #define later by 토크나이저
    scalar_swiglue: int = 256
    ffn_dim_multiplier: Optional[float] = None

    use_cache: bool = True
    max_batch_size: int = 2
    max_sequence_length: int = 2048




class LLamaAttention(nn.Module):
    def __init__(self, args = AttentionConfig):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.use_cache = args.use_cache

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.head_dim = self.hidden_dim // self.num_attention_heads
        assert (
            self.head_dim * self.num_attention_heads == self.hidden_dim
        ), "hidden_dim must be divisible by num_attention_heads"

    def forward(self, x, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        output = self.o_proj(attn_output)

        return output
    
