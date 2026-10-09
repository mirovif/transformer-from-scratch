import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from AddNorm import AddNorm
from FeedForward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, D, heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(D, heads)
        self.norm1 = AddNorm(D)
        self.feed_forward = FeedForward(D, d_ff)
        self.norm2 = AddNorm(D)

    def forward(self, x, mask=None):

        attention_output = self.attention(x, x, x, mask)

        x = self.norm1(x, attention_output)
        ff_output = self.feed_forward(x)

        x = self.norm2(x, ff_output)

        return x