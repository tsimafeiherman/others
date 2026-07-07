import torch
from torch import nn

from config import ModelConfig

class ROPE(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        
        self.model_model_cfg = model_cfg
        self.max_seq_length = model_cfg.n_ctx
        self.theta = model_cfg.theta
        self.head_dim = model_cfg.head_dim
        
        # Создаём индексы пар: [0, 1, 2, ..., head_dim//2 - 1].
        # Почему head_dim // 2: У нас head_dim размерностей, они разбиты на пары.
        # Каждая пара вращается со своей частотой. Значит, нам нужно ровно head_dim // 2 разных частот.
        indexes = torch.arange(0, model_cfg.head_dim // 2) # [d_head // 2]
        
        # Вычисляем частоты θᵢ по формуле из статьи
        # Чтобы покрыть все масштабы расстояний — от локальных (высокая частота) до глобальных (низкая частота).
        # Это даёт модели многомасштабное представление позиций.
        freqs = self.theta ** (-2 * indexes / self.head_dim) # [d_head // 2]
        
        # Создаём все возможные позиции токенов: [0, 1, 2, ..., max_seq_length-1].
        positions_id = torch.arange(0, self.max_seq_length).float() # [max_seq_length]
        
        # positions_id.unsqueeze(-1): [max_seq_length, 1] — каждая позиция в своей строке
        # freqs.unsqueeze(0): [1, head_dim // 2] — каждая частота в своём столбце
        # Результат: [max_seq_length, head_dim // 2] — матрица углов
        # Что в ячейке [m, i]: угол m × θᵢ — на сколько радиан нужно повернуть i-ю пару для токена на позиции m.
        idx_theta = positions_id.unsqueeze(-1) * freqs.unsqueeze(0) # (max_seq_length, 1) * (1, d_head // 2) = (max_seq_length, d_head // 2)
        
        # Просто берём косинус и синус от каждого угла.
        # Зачем: Формула поворота вектора (x₀, x₁) на угол φ:
        cos = idx_theta.cos() # (max_seq_length, d_head // 2)
        sin = idx_theta.sin() # (max_seq_length, d_head // 2)
        
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1) # (max_seq_length, d_head)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1) # (max_seq_length, d_head)
        
        cos = cos.view(1, self.max_seq_length, 1, self.head_dim) # (1, max_seq_length, 1, d_head)
        sin = sin.view(1, self.max_seq_length, 1, self.head_dim) # (1, max_seq_length, 1, d_head)

        # Это не обучаемые параметры, но они сохраняются с моделью и ездят на GPU
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    @staticmethod
    def rotate_neg_vector(x):
        # x: (bs, seq_len, num_heads, d_head)
        # На входе x = [x1, x2, x3, x4, ... x_{n-1}, x_n]
        # На выходе x' = [-x2, x1, -x4, x3, ..., -x_n, x_{n-1}]
        
        # Зачем это нужно:
            # Формула поворота:
            # y₀ = x₀*cos(φ) - x₁*sin(φ) = x₀*cos(φ) + (-x₁)*sin(φ)
            # y₁ = x₁*cos(φ) + x₀*sin(φ)
        
        x_out = torch.empty_like(x) # (bs, seq_len, num_heads, head_dim)
        bs, seq_len, num_heads, head_dim = x_out.size()
        
        even_idx = torch.arange(0, head_dim, step=2)  
        odd_idx = torch.arange(1, head_dim, step=2)  

        x_out[..., even_idx] = -x[...,odd_idx]
        x_out[..., odd_idx] = x[..., even_idx]
        
        return x_out
        
    def forward(self, x):
        # x: (bs, seq_len, num_heads, d_head)
        
        seq_len = x.size(1)
        
        x_rot = self.rotate_neg_vector(x)
        
        return x * self.cos[:, :seq_len] + x_rot * self.sin[:, :seq_len]