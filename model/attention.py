import torch
import torch.nn as nn


class DynamicHierarchicalAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(DynamicHierarchicalAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.char_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.block_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.global_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.merge_layer = nn.Linear(d_model, 1)
        self.global_merge_layer = nn.Linear(d_model, 1)

    def merge_tokens(self, embeddings, grad_layer):
        embeddings.requires_grad_()
        grad = torch.autograd.grad(embeddings.sum(), embeddings, create_graph=True)[0]
        importance_scores = grad_layer(grad).squeeze(-1)

        cumulative_scores = torch.cumsum(importance_scores, dim=1)
        block_boundaries = (cumulative_scores % 1 < cumulative_scores.roll(1, dims=1) % 1).float()
        block_ids = torch.cumsum(block_boundaries, dim=1)

        max_blocks = int(block_ids.max().item())
        merged_embeddings = []
        block_counts = []

        for i in range(1, max_blocks + 1):
            block_mask = (block_ids == i).float().unsqueeze(-1)
            block_sum = (embeddings * block_mask).sum(dim=1)
            block_count = block_mask.sum(dim=1)
            block_avg = block_sum / (block_count + 1e-10)
            merged_embeddings.append(block_avg)
            block_counts.append(block_count)

        merged_embeddings = torch.stack(merged_embeddings, dim=1)
        block_counts = torch.stack(block_counts, dim=1)

        return merged_embeddings, importance_scores, block_ids, block_counts

    def forward(self, x, mask=None):
        # 1. 字符级自注意力
        char_out, char_weights = self.char_attention(x, x, x, attn_mask=mask)

        # 2. 合并字符为块
        block_embeddings, char_importance, char_block_ids, char_block_counts = self.merge_tokens(char_out,
                                                                                                 self.merge_layer)

        # 3. 块级自注意力
        block_out, block_weights = self.block_attention(block_embeddings, block_embeddings, block_embeddings)

        # 4. 合并块为全局块
        global_embeddings, block_importance, global_block_ids, global_block_counts = self.merge_tokens(block_out,
                                                                                                       self.global_merge_layer)

        # 5. 全局自注意力
        global_out, global_weights = self.global_attention(global_embeddings, global_embeddings, global_embeddings)

        # 6. 将全局表示扩展回原始序列长度
        expanded_global_out = global_out.repeat_interleave(global_block_counts.long(), dim=1)
        expanded_global_out = expanded_global_out.repeat_interleave(char_block_counts.long(), dim=1)

        return expanded_global_out, (char_weights, block_weights, global_weights,
                                     char_importance, char_block_ids,
                                     block_importance, global_block_ids)
