import math
import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from InputEmbedding import InputEmbedding




class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        D,
        heads,
        d_ff,
        N,
        max_len,
        dropout
    ):
        super().__init__()

        self.src_input = InputEmbedding(src_vocab, D, max_len, dropout)
        self.tgt_input = InputEmbedding(tgt_vocab, D, max_len, dropout)

        self.encoder = Encoder(D, heads, d_ff, N)
        self.decoder = Decoder(D, heads, d_ff, N)
        self.lin = nn.Linear(D, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        tgt = self.tgt_input(tgt)
        src = self.src_input(src)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        l = self.lin(decoder_output)

        return l

if __name__ == "__main__":
    model = Transformer(
        src_vocab=20,
        tgt_vocab=25,
        D=8,
        heads=2,
        d_ff=32,
        N=2,
        max_len=50,
        dropout=0.1
    )

    src = torch.randint(0, 20, (2, 5))
    tgt = torch.randint(0, 25, (2, 4))

    output = model(src, tgt)

    print("src:", src.shape)
    print("tgt:", tgt.shape)
    print("output:", output.shape)