import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=1):
        super(CausalConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                              groups=channels)

        # 创建一个因果掩码
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1:, :] = 0
        mask[kernel_size // 2, kernel_size // 2 + 1:] = 0
        self.register_buffer('mask', mask.view(1, 1, kernel_size, kernel_size))

    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)


class DynamicConvolutionalBlockedAttention(nn.Module):
    def __init__(self, d_model, nhead, block_size=16):
        super(DynamicConvolutionalBlockedAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.block_size = block_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.conv_layer = CausalConv2d(nhead, kernel_size=3, stride=1)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, conv_times=1):
        B, L, C = x.size()
        H = self.nhead

        # 计算Q, K, V
        q = self.query(x).view(B, L, H, -1).transpose(1, 2)
        k = self.key(x).view(B, L, H, -1).transpose(1, 2)
        v = self.value(x).view(B, L, H, -1).transpose(1, 2)

        # 初始块状注意力计算
        block_size = min(self.block_size, L)
        num_blocks = (L + block_size - 1) // block_size

        attn_scores = torch.zeros(B, H, L, L, device=x.device)

        for i in range(num_blocks):
            for j in range(i + 1):  # 只计算下三角部分
                start_i, start_j = i * block_size, j * block_size
                end_i, end_j = min((i + 1) * block_size, L), min((j + 1) * block_size, L)

                q_block = q[:, :, start_i:end_i]
                k_block = k[:, :, start_j:end_j]

                scores = torch.matmul(q_block, k_block.transpose(-1, -2)) / (C // H) ** 0.5
                attn_scores[:, :, start_i:end_i, start_j:end_j] = scores

        # 应用因果掩码
        causal_mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # 进行卷积操作
        attn_scores = attn_scores.transpose(0, 1)  # [H, B, L, L]
        for _ in range(conv_times):
            attn_scores = self.conv_layer(attn_scores)

        attn_scores = attn_scores.transpose(0, 1)  # [B, H, L, L]

        # 应用softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 应用注意力
        out = torch.matmul(attn_probs, v)

        # 重塑并投影输出
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        out = self.out_proj(out)

        return out

