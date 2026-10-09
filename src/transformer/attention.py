import torch
import torch.nn as nn
import math
from .masks import canonical_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, D, heads):
        super().__init__()
        self.D = D
        self.heads = heads


        if heads <= 0 or D <= 0 or D % heads != 0:
            raise ValueError("D % heads != 0")
        self.d_k = D // heads


        self.Q = nn.Linear(D, D)
        self.K = nn.Linear(D, D)
        self.V = nn.Linear(D, D)

        self.W_o = nn.Linear(D, D)

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]
        T_q = query.shape[1]
        T_k = key.shape[1]

        query_x = self.Q(query)
        key_x = self.K(key)
        value_x = self.V(value)

        query_x = query_x.view(B, T_q, self.heads, self.d_k)
        query_x = query_x.transpose(1, 2)

        key_x = key_x.view(B, T_k, self.heads, self.d_k)
        key_x = key_x.transpose(1, 2)

        value_x = value_x.view(B, T_k, self.heads, self.d_k)
        value_x = value_x.transpose(1, 2)

        key_x = key_x.transpose(2, 3)

        numerator = query_x @ key_x

        scores = numerator / math.sqrt(self.d_k)

        allowed = None
        if mask is not None:
            allowed = canonical_mask(mask).to(device=scores.device)
            try:
                allowed = torch.broadcast_to(allowed, scores.shape)
            except RuntimeError as error:
                raise ValueError("Mask cannot broadcast to (B, heads, T_q, T_k)") from error
            # Finite minimum plus zeroing handles completely masked rows without NaN.
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        new_scores = scores.softmax(dim=-1)
        if allowed is not None:
            new_scores = new_scores.masked_fill(~allowed, 0.0)
        self_attention = new_scores @ value_x
        # форма: (B, heads, T, d_k)
        self_attention = self_attention.transpose(1, 2)
        self_attention = self_attention.reshape(B, T_q,self.D)

        output = self.W_o(self_attention)

        return output


if __name__ == "__main__":
    attention = MultiHeadAttention(D=8, heads=2)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)

    output = attention(query, key, value)
    print(output.shape)