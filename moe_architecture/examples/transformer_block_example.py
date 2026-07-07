import torch
from torch import nn

from config import ModelConfig
from rmsnorm import RMSNorm
from attention import ROPE_MHSA
from examples.moe_layer_example import MoELayer

class TransformerBlock(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ln_1 = RMSNorm(model_cfg)
        self.attention = ROPE_MHSA(model_cfg)
        self.ln_2 = RMSNorm(model_cfg)
        self.moe = MoELayer(model_cfg)
        
    def forward(self, x, mask):
        # x: (bs, seq_len, embed_dim)
        
        x = x + self.attention(self.ln_1(x), mask)
        temp, aux_loss = self.moe(self.ln_2(x))
        x = x + temp
        
        return x, aux_loss