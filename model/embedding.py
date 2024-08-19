import torch
from torch import nn




class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        return self.position_encoding(self.embedding(x))
