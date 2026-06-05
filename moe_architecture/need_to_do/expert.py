"""
    Класс ExpertMLP.

    Внутри: Linear(d_model, d_mlp), GELU(), Linear(d_mlp, d_model).

    Это точно такой же MLP, как в обычном трансформере, только будет использоваться внутри MoE.

    Подсказки:
    1. Размерность входа: d_model (например, 256).
    2. Размерность скрытого слоя: d_mlp = mlp_mult * d_model (например, 4 * 256 = 1024).
    3. Порядок: Linear(d_model, d_mlp) → GELU → Linear(d_mlp, d_model).
    4. Вход x имеет форму [N, d_model], где N — количество токенов, назначенных этому эксперту.
    5. Residual connection (x + ...) НЕ нужен — он будет снаружи, в TransformerBlock.
"""

import torch
from torch import nn
from config import ModelConfig


class ExpertMLP(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # TODO: Создайте первый линейный слой (d_model → d_mlp)
        # Подсказка: d_mlp = model_cfg.mlp_mult * model_cfg.d_model
        self.in_linear = ...  # ваш код здесь
        
        # TODO: Создайте функцию активации GELU
        self.gelu = ...  # ваш код здесь
        
        # TODO: Создайте второй линейный слой (d_mlp → d_model)
        self.out_linear = ...  # ваш код здесь
        
    def forward(self, x):
        # x: [N, d_model] — пакет токенов, назначенных этому эксперту
        # N может быть разным для разных экспертов и разных батчей
        # Порядок: in_linear → GELU → out_linear
        # Возвращает: [N, d_model]
        
        # TODO: Пропишите forward pass
        return ...  # ваш код здесь