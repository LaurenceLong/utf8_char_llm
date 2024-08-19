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

        print(f"x shape: {x.shape}, pe shape: {self.pe.shape}, max_seq_len: {max_seq_len}")
        print(f"x dtype: {x.dtype}, pe dtype: {self.pe.dtype}")
        print(f"x device: {x.device}, pe device: {self.pe.device}")
        # 打印 x 的统计信息
        # 将 x 移到 CPU 并分离计算图
        x_cpu = x.detach().cpu()

        try:
            print(f"x statistics:")
            print(f"  Min: {x_cpu.min().item()}")
            print(f"  Max: {x_cpu.max().item()}")
            print(f"  Mean: {x_cpu.mean().item()}")
            print(f"  Std: {x_cpu.std().item()}")
        except Exception as e:
            print(f"Error calculating statistics: {e}")

        # 检查是否有 NaN 或 Inf 值
        try:
            nan_count = torch.isnan(x_cpu).sum().item()
            inf_count = torch.isinf(x_cpu).sum().item()
            print(f"NaN count: {nan_count}, Inf count: {inf_count}")
        except Exception as e:
            print(f"Error checking for NaN/Inf: {e}")

        # 打印 x 的一小部分样本
        try:
            print("Sample of x (first 3 batches, first 5 sequence elements, first 5 features):")
            print(x_cpu[:3, :5, :5])
        except Exception as e:
            print(f"Error printing sample: {e}")

        return x + self.pe[:, :max_seq_len, :]
