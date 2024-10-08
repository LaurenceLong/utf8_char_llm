import torch.nn as nn
import torch.nn.functional as F


class OutputLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)
