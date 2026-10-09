import math
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .embedding import InputEmbedding
from .masks import canonical_mask, padding_mask, decoder_mask




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
        dropout,
        pad_id=0
    ):
        super().__init__()

        self.pad_id = pad_id
        self.src_input = InputEmbedding(src_vocab, D, max_len, dropout, pad_id)
        self.tgt_input = InputEmbedding(tgt_vocab, D, max_len, dropout, pad_id)

        self.encoder = Encoder(D, heads, d_ff, N)
        self.decoder = Decoder(D, heads, d_ff, N)
        self.lin = nn.Linear(D, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src.ndim != 2 or tgt.ndim != 2 or src.shape[0] != tgt.shape[0]:
            raise ValueError("Source and target must have shapes (B, T_src) and (B, T_tgt)")
        source_allowed = padding_mask(src, self.pad_id)
        target_allowed = decoder_mask(tgt, self.pad_id)
        if src_mask is not None:
            extra_source = canonical_mask(src_mask).to(src.device)
            if extra_source.shape[-2] != 1:
                raise ValueError("src_mask must mask keys only: (B, 1, 1, T_src)")
            source_allowed = source_allowed & extra_source
        if tgt_mask is not None:
            target_allowed = target_allowed & canonical_mask(tgt_mask).to(tgt.device)
        src_mask, tgt_mask = source_allowed, target_allowed
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