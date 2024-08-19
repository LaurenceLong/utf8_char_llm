import torch.nn as nn

from .attention import HierarchicalSelfAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(DecoderLayer, self).__init__()
        self.self_attn = HierarchicalSelfAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, attention_weights = self.self_attn(x, mask)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        return x, attention_weights



class DecoderStack(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        attention_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            attention_weights.append(weights)
        return x, attention_weights

