import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .add_norm import AddNorm
from .feed_forward import FeedForward




class DecoderBlock(nn.Module):
    def __init__(self, D, heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(D, heads)
        self.norm1 = AddNorm(D)
        self.cross_attention = MultiHeadAttention(D, heads)
        self.norm2 = AddNorm(D)
        self.feed_forward = FeedForward(D, d_ff)
        self.norm3 = AddNorm(D)

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        attention_output = self.attention(x, x, x, self_mask)
        x = self.norm1(x, attention_output)

        cross_attention_output = self.cross_attention(x, encoder_output, encoder_output, cross_mask)

        x = self.norm2(x, cross_attention_output)
        ff_output = self.feed_forward(x)

        x = self.norm3(x, ff_output)


        return x