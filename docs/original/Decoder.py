import torch
import torch.nn as nn
from Decoder_Block import DecoderBlock



class Decoder(nn.Module):
    def __init__(self, D, heads, d_ff, N):
        super().__init__()

        self.decoder = nn.ModuleList([
            DecoderBlock(D, heads, d_ff)
            for _ in range(N)
        ])

    def forward(self,x, encoder_output, self_mask=None, cross_mask=None):
        for block in self.decoder:
            x = block(x, encoder_output, self_mask, cross_mask)

        return x