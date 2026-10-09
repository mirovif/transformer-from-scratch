import torch.nn as nn



class AddNorm(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(D)
        # создать LayerNorm для размерности D

    def forward(self, x, sublayer_output):
        result = self.LayerNorm(x + sublayer_output)
        return result


