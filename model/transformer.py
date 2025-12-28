import torch
import torch.nn as nn
from model.attention import LLamaAttention, AttentionConfig
from model.layer_norm import LLamaRMSNorm
from model.embeddings import LLamaEmbedding
from model.rope import precompute_freqs_cis, reshape_for_broadcasting, rotary_emb
from model.feedforward import FeedForward
from typing import Optional, Tuple

class LLamaTransformerBlock(nn.Module):
    def __init__(self,layer_id:int, args: AttentionConfig):
        """
        layer_id: layer identifier
        args: config
    
        """
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = args.num_attention_heads
        self.hidden_dim = args.hidden_dim
        self.attention = LLamaAttention(args)
        self.ffn = FeedForward(
            dim = args.hidden_dim,
            hidden_dim= 4*args.hidden_dim,
            scale_to_hidden_dim = args.scale_to_hidden_dim,
            ffn_dim_multiplier = args.ffn_dim_multiplier,
        )
        self.norm = LLamaRMSNorm(args.hidden_dim, eps = args.norm_eps)
        self.ffn_norm = LLamaRMSNorm(args.hidden_dim, eps = args.norm_eps)


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        
        mid_output = x + self.attention(
            self.norm(x), start_pos, freqs_cis, mask
        )
        out = mid_output + self.ffn(
            self.ffn_norm(mid_output)
        )
        
        return out
    

class LLamaTransformer(nn.Module):
    def __init__(self,args: AttentionConfig):
        super().__init__()
        self.paramas = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.num_hidden_layers

        self.token_embedding = LLamaEmbedding(
            vocab_size = args.vocab_size,
            embedding_dim = args.hidden_dim
        )
        self.layers = nn.ModuleList()
        self.norm = LLamaRMSNorm(
            hidden_size = args.hidden_dim,
            eps = args.norm_eps
        )
        for layer_id in range(self.n_layers):
            self.layers.append(
                LLamaTransformerBlock(
                    layer_id = layer_id,
                    args = args
                )
            )
        self.output = nn.Linear(
            args.hidden_dim,
            args.vocab_size,
            bias = False
        )
        self.freqs_cis = precompute_freqs_cis(
            self.args.hidden_dim //  self.args.num_attention_heads, self.args.max_seq_len * 2
        )
    @torch.inference_mode()
    def forward(self, tokens:torch.Tensor, start_pos:int):
        """
        tokens: (batch_size, seq_len) input token idices
        start_pos: position to start for attention caching
        """
        batch_sz,seq_len = tokens.shape
        x = self.token_embedding(tokens)  # (batch_size, seq_len, hidden_dim)
        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        mask = None  # For simplicity, no mask is applied here.
        
        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len),
                float('-inf'),
                device = tokens.device
            )
            mask = torch.triu(mask, diagonal=1)

            mask = torch.hstack(
                [torch.zeros((seq_len, start_pos), device = tokens.device), mask]
            ).type_as(x)

        for layer in self.layers:
            x = layer(
                x,
                start_pos,
                freqs_cis,
                mask
            )
        x = self.norm(x)
        output = self.output(x).float()
        return output


