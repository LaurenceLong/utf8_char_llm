import math

import torch
from torch import nn

import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        max_seq_len = x.size(1)
        if max_seq_len > self.max_len:
            raise ValueError(f"Input sequence length ({max_seq_len}) exceeds maximum length ({self.max_len})")
        return x + self.pe[:, :max_seq_len, :]
