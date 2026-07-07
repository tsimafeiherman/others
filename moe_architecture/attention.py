import torch
from torch import nn

from config import ModelConfig
from rmsnorm import RMSNorm
from rope import ROPE

class ROPE_MHSA(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):       
        """  
            Класс Attention с ROPE.

            Размерности:
            проекции [d_model, n_heads * d_head], затем разделение на головы через .view и .permute.

            Метод apply_causal_mask(attn_scores, mask):
                объединяет нижнюю треугольную маску (torch.tril) и padding-маску (переданную как mask [batch, seq_len]).
                Padding-маску нужно привести к [batch, 1, 1, seq_len].

            Не забудь деление на sqrt(d_head) перед softmax.

            Возвращает выход [batch, seq_len, d_model].
        """
        super().__init__(*args, **kwargs)
        
        self.num_heads = model_cfg.n_heads
        self.head_dim = model_cfg.head_dim
        
        self.qkv_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model * 3)
        
        self.rope = ROPE(model_cfg)
        
        self.final_norm = RMSNorm(model_cfg)
        self.final_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model)
        
        self.register_buffer("IGNORE", torch.tensor(float("-inf")))
        
    def _apply_causal_mask(self, attn_scores, mask):
        
        # attn_scores: (bs, num_heads, seq_len, seq_len)
        # mask: (bs, seq_len)
        
        seq_len = attn_scores.shape[-1]
        causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
        
        padding_mask = mask.unsqueeze(1).unsqueeze(2) # (bs, 1, 1, seq_len)
        
        combined = causal_mask * padding_mask # (1, 1, seq_len, seq_len) * (bs, 1, 1, seq_len) = (bs, 1, seq_len, seq_len)
        
        # 2 способа применять маску
        attn_scores = attn_scores.masked_fill(combined == 0, float("-inf"))
        # attn_scores = torch.where(combined.bool(), attn_scores, float("-inf"))
        
        return attn_scores
        
    def forward(self, x , mask=None):
        # x: (bs, seq_len, embed_dim)
        
        bs, seq_len, _ = x.size()
        
        qkv = self.qkv_proj(x) # (bs, seq_len, embed_dim * 3)
        qkv = qkv.view(bs, seq_len, 3, self.num_heads, self.head_dim) # (bs, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2,0,3,1,4).contiguous() # (3, bs, num_heads, seq_len, head_dim)
        
        Q = qkv[0] # (bs, num_heads, seq_len, head_dim)
        K = qkv[1] # (bs, num_heads, seq_len, head_dim)
        V = qkv[2] # (bs, num_heads, seq_len, head_dim)
        
        Q = self.rope(Q)
        K = self.rope(K)
        
        attn_scores = Q @ K.transpose(-1, -2) / self.head_dim ** 0.5  # (bs, num_heads, seq_len, seq_len)
        
        attn_scores = self._apply_causal_mask(attn_scores, mask)
        
        probs = torch.softmax(attn_scores, dim=-1) # (bs, num_heads, seq_len, seq_len)
        out = probs @ V # (bs, num_heads, seq_len, head_dim)
        
        out = out.permute(0,2,1,3).contiguous()  # (bs, seq_len ,num_heads * head_dim)
        out = out.view(bs, seq_len, -1)  # (bs, seq_len, embed_dim)
        
        out = self.final_norm(out)
        out = self.final_proj(out)
        
        return out