import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig

class Router(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model_cfg = model_cfg
        
        self.gate = nn.Linear(model_cfg.d_model, model_cfg.n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)
    
    def calc_load_balancing_loss(self, top_k_indices, router_logits):
        # top_k_indices: (bs, seq_len, k_experts)
        # router_logits: (bs, seq_len, n_experts)
        
        top_k_indices = top_k_indices.view(-1, self.model_cfg.k_experts) # (bs * seq_len, k_experts)
        router_logits = router_logits.view(-1, self.model_cfg.n_experts) # (bs * seq_len, k_experts)
        
        mask = F.one_hot(top_k_indices, num_classes=self.model_cfg.n_experts)
        mask = torch.sum(mask, dim=1) # (bs * seq_len, n_experts)
        mask = (mask > 0).float() # (bs * seq_len, n_experts)
        
        # реальная доля токенов у каждого эксперта
        fraction = mask.mean(dim=0) # (n_experts)
        
        # средняя "желаемая" вероятность эксперта
        router_probs = torch.softmax(router_logits, dim=-1) # (bs * seq_len, n_experts)
        gates_mean = router_probs.mean(dim=0) # (n_experts) 
        
        aux_loss = torch.sum(fraction * gates_mean, dim=-1) * self.model_cfg.n_experts
        
        return aux_loss   
    
    def forward(self, x):
        # x: (bs, seq_len, embed_dim)
        
        logits = self.gate(x) # (bs, seq_len, n_experts)
        
        probs = torch.softmax(logits, dim=-1) # (bs, seq_len, n_experts)
        top_k_probs, top_k_indices = torch.topk(probs, self.model_cfg.k_experts, dim=-1) # (bs, seq_len, k_experts)
        
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices, logits