import torch
from torch import nn

from config import ModelConfig

class ExpertMLP(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.in_linear = nn.Linear(model_cfg.d_model, model_cfg.mlp_mult * model_cfg.d_model)
        self.gelu = nn.GELU()
        self.out_linear = nn.Linear(model_cfg.mlp_mult * model_cfg.d_model, model_cfg.d_model)
        
    def forward(self, x):
        # x: (N, embed_dim)
        
        return self.out_linear(self.gelu(self.in_linear(x)))