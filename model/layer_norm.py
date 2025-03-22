import torch
import torch.nn as nn

class LLamaRMSNorm(nn.Module):
    def __init__(self,hidden_size, eps = 1e-6,bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.isbias = bias

    def forward(self, x):
        """
        RMS Norm
            y = ( (x - mean) / sqrt(variance + eps)) * weight + bias
        
        LLama RMS Norm
            y = ( x / sqrt(variance + eps)) * weight


        x: 인풋 텐서 (batch_size, seq_len, hidden_size)
        return 정규화된 텐서
        """
        variance = torch.mean(x ** 2, dim=-1, keepdim=True)
        x = x * (self.weight / torch.sqrt(variance + self.eps))
        x *= self.weight
        return x
    

    