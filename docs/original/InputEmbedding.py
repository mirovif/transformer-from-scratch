import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, V, D, max_len, dropout):
        super().__init__()

        self.D = D
        self.embedding = nn.Embedding(V, D)
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        denominator = 10000 ** (
            torch.arange(0, D, 2, dtype=torch.float32) / D
        )

        angles = positions / denominator

        pe = torch.zeros(max_len, D)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, :D // 2])

        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.shape[1]

        x = self.embedding(x) * math.sqrt(self.D)
        positional_encoding = self.pe[:T].unsqueeze(0)

        x = x + positional_encoding
        return self.dropout(x)