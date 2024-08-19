import torch.nn as nn


class BlockwiseTokenMerger(nn.Module):
    def __init__(self):
        super(BlockwiseTokenMerger, self).__init__()

    def forward(self, embeddings, attention_gradients):
        # 简化实现，实际上这里应该基于注意力梯度动态合并token
        return embeddings