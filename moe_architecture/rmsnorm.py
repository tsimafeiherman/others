import torch
from torch import nn

from config import ModelConfig

"""
    Класс RMSNorm с параметром w (gamma), без bias.

    Формула: x / sqrt(mean(x^2) + eps) * w.

    Размер: w имеет shape [d_model], нормализация по последней размерности.
"""

class RMSNorm(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Инициализируем единицами, чтобы все были на одном уровне сразу и после первого прохода модель сама двигала веса
        # Если инициализировать нулями, то слой просто обучаться не будет, так как и градиент 0
        # Если инициалзирвоать нормальным распределением, то изначально модели прийдется корректировать данный шум
        self.W = nn.Parameter(torch.ones((model_cfg.d_model)), requires_grad=True)
        
    def forward(self, x):
        # x: (bs, seq_len, d_model)
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6) # (bs, seq_len, 1)
        x_norm = x / rms # (bs, seq_len, d_model)
        
        return x_norm * self.W
        
        