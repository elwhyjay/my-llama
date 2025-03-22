import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LLamaEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


    def forward(self, x):
        return self.embedding(x)