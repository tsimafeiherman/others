import torch
from torch import nn

from config import ModelConfig
from examples.transformer_block_example import TransformerBlock
from rmsnorm import RMSNorm

class MoETransformer(nn.Module):
    def __init__(self, model_cfg:ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.embed = nn.Embedding(num_embeddings=model_cfg.vocab_size, embedding_dim=model_cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(model_cfg) for _ in range(model_cfg.layers)])
        
        self.ln_final = RMSNorm(model_cfg)
        self.final_proj = nn.Linear(model_cfg.d_model, model_cfg.vocab_size)
        
    def forward(self, input_ids, mask):
        # input_ids: (bs, seq_len)
        
        x = self.embed(input_ids) # (bs, seq_len, embed_dim)
        
        total_aux = 0.0
        for layer in self.layers:
            x, aux = layer(x, mask)
            total_aux += aux
            
        x = self.ln_final(x) # (bs, seq_len, embed_dim)
        logits = self.final_proj(x) # (bs, seq_len, vocab_size)
        
        return logits, total_aux
        