import torch
from torch import nn

from config import ModelConfig
from examples.expert_example import ExpertMLP
from examples.router_example import Router

class MoELayer(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_cfg = model_cfg
        
        self.experts = nn.ModuleList([ExpertMLP(model_cfg) for _ in range(self.model_cfg.n_experts)])
        self.router = Router(model_cfg)
        
    def forward(self, x):
        
        # x: (bs, seq_len, embed_dim)
        
        # top_k_probs: (bs, seq_len, k_experts)
        # top_k_indices: (bs, seq_len, k_experts)
        # logits: (bs, seq_len, n_experts)
        top_k_probs, top_k_indices, logits = self.router(x)
        
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.model_cfg.n_experts):
            
            # маска токенов для эксперта, которым он нужен
            expert_mask = (top_k_indices == expert_idx).any(dim=-1) # (bs, seq_len)

            if expert_mask.any():
                
                # сами нужные токены
                expert_input = x[expert_mask] # (N, embed_dim)
                
                expert_output = self.experts[expert_idx](expert_input) # (N, d_model)
                
                # Пример:
                # top_k_indices[токен] = [3, 5] (выбраны эксперты 3 и 5)
                # Для i = 3: (top_k_indices == 3) даст [True, False]
                # Для i = 5: (top_k_indices == 5) даст [False, True]
                # Для i = 0: (top_k_indices == 0) даст [False, False]
                token_pos_mask = (top_k_indices == expert_idx) # (bs , seq_len, k_experts)
                
                # Пример:
                # top_k_probs[токен] = [0.7, 0.3]
                # Для i = 3: [True, False].float() * [0.7, 0.3] = [1, 0] * [0.7, 0.3] = [0.7, 0]
                # Для i = 5: [False, True].float() * [0.7, 0.3] = [0, 1] * [0.7, 0.3] = [0, 0.3]
                expert_weights = token_pos_mask.float() * top_k_probs # (bs , seq_len, k_experts) * (bs , seq_len, k_experts) = (bs , seq_len, k_experts)
                
                # Пример
                # Для i = 3: [0.7, 0].sum() = 0.7
                # Для i = 5: [0, 0.3].sum() = 0.3
                expert_weights = expert_weights.sum(dim=-1) # (bs, seq_len)
                filttered_expert_weights = expert_weights[expert_mask] # (N) - колво токенов для 1 эксперта
                filttered_expert_weights = filttered_expert_weights.unsqueeze(-1) # (N, 1)
                
                output[expert_mask] += expert_output * filttered_expert_weights
        
        aux_loss = torch.tensor(0.0)
        if self.training:
            aux_loss = self.router.calc_load_balancing_loss(top_k_indices, logits)
            
            return output, aux_loss
        
        return output, aux_loss
                