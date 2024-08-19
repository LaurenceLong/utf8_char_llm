import torch.nn as nn


class HierarchicalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(HierarchicalSelfAttention, self).__init__()
        self.char_attention = nn.MultiheadAttention(d_model, nhead)
        self.block_attention = nn.MultiheadAttention(d_model, nhead)
        self.global_attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x, mask=None):
        char_out, char_weights = self.char_attention(x, x, x, attn_mask=mask)
        block_out, block_weights = self.block_attention(char_out, char_out, char_out, attn_mask=mask)
        global_out, global_weights = self.global_attention(block_out, block_out, block_out, attn_mask=mask)
        return global_out, (char_weights, block_weights, global_weights)
