import torch
import torch.nn as nn
from .encoder_block import EncoderBlock




class Encoder(nn.Module):
    def __init__(self, D, heads, d_ff, N):
        super().__init__()

        self.encoder = nn.ModuleList([
            EncoderBlock(D, heads, d_ff)
            for _ in range(N)
            ])

    def forward(self, x, mask=None):
        for block in self.encoder:
            x = block(x, mask)

        return x

if __name__ == "__main__":
    encoder = Encoder(D=8, heads=2, d_ff=32, N=2)

    x = torch.randn(2, 4, 8)
    output = encoder(x)

    print(output.shape)