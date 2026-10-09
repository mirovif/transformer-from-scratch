"""Boolean masks: True allows attention; False blocks it."""
import torch

def canonical_mask(mask):
    if mask.dtype != torch.bool:
        raise TypeError('Masks must be boolean; True means an allowed attention position.')
    if mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0)  # shared (T_q, T_k)
    if mask.ndim == 3:
        return mask.unsqueeze(1)  # (B, T_q, T_k)
    if mask.ndim == 4:
        return mask
    raise ValueError('Expected mask with 2, 3 or 4 dimensions.')

def padding_mask(tokens, pad_id=0):
    return (tokens != pad_id).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)

def causal_mask(length, device=None):
    return torch.ones(length, length, dtype=torch.bool, device=device).tril().unsqueeze(0).unsqueeze(0)

def decoder_mask(tokens, pad_id=0):
    return causal_mask(tokens.shape[1], tokens.device) & padding_mask(tokens, pad_id)
