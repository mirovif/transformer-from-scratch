import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, D, d_ff):
        super().__init__()
        self.lin1 = nn.Linear(D, d_ff)
        self.Relu = nn.ReLU()
        self.lin2 = nn.Linear(d_ff, D)

    def forward(self, x):
        x = self.lin1(x)
        x = self.Relu(x)
        x = self.lin2(x)
        return x