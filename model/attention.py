import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from model.rope import rotary_emb
from config.model_config import LlamaConfig

def fast_repeat_interleave(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    batch_sz, seq_len, n_kv_heads, head_dim = tensor.shape
    if repeats == 1:
        return tensor
    return (
        tensor[:, :, :, None, :]
        .expand(batch_sz, seq_len, n_kv_heads, repeats, head_dim)
        .reshape(batch_sz, seq_len, n_kv_heads * repeats, head_dim)
    )


class LLamaAttention(nn.Module):
    def __init__(self, args: LlamaConfig):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # For GQA
        self.num_rep = self.num_attention_heads // self.num_key_value_heads
        
        self.use_cache = args.use_cache
        self.head_dim = self.hidden_dim // self.num_attention_heads
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        assert (
            self.head_dim * self.num_attention_heads == self.hidden_dim
        ), "hidden_dim must be divisible by num_attention_heads"

        self.cache_k = torch.zeros(
            args.max_batch_size,
            args.max_sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).cuda()
        self.cache_v = torch.zeros(
            args.max_batch_size,
            args.max_sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).cuda()
    def forward(self, x, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        q,k = rotary_emb(q, k, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(q)
        self.cache_v = self.cache_v.to(q)

        self.cache_k[:batch_size, start_pos : start_pos + seq_len, :, :] = k
        self.cache_v[:batch_size, start_pos : start_pos + seq_len, :, :] = v

        keys = self.cache_k[:batch_size, : start_pos + seq_len, :, :]
        values = self.cache_v[:batch_size, : start_pos + seq_len, :, :]

        #k = keys.repeat_interleave(self.num_rep, dim=2)
        #v = values.repeat_interleave(self.num_rep, dim=2)
        keys = fast_repeat_interleave(keys, self.num_rep)
        values = fast_repeat_interleave(values, self.num_rep)

        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        attn_scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            attn_scores = attn_scores+mask # (batch_size, num_heads, seq_len, cache_len+seq_len)
        attn_probs = F.softmax(attn_scores, dim=-1).type_as(q)
        output = torch.matmul(attn_probs, values)  # (batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len,-1)
        output = self.o_proj(output)   

        return output
    
