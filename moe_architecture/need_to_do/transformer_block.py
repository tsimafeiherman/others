"""
    Класс TransformerBlock (MoE Transformer Block).

    Один блок трансформера с Mixture of Experts вместо обычного MLP.

    Компоненты:
        - ln_1: RMSNorm — нормализация перед Attention
        - attention: Multi-Head Self-Attention с RoPE
        - ln_2: RMSNorm — нормализация перед MoE
        - moe: MoELayer — слой Mixture of Experts (роутер + эксперты)

    forward(x, mask):
        1. x = x + attention(ln_1(x), mask)    — Attention с residual connection
        2. moe_out, aux_loss = moe(ln_2(x))     — MoE слой
        3. x = x + moe_out                       — residual connection
        4. Вернуть (x, aux_loss)

    Это Pre-LN стиль (нормализация ДО подслоя, а не после), как в GPT-2/GPT-3.

    Важно:
        - Residual connection (x + ...) — обязателен для глубоких сетей
        - Возвращаем кортеж (x, aux_loss) — aux_loss нужен для балансировки экспертов
        - Маску передаём только в attention (MoE про неё не знает)

    Размерности:
        x:          [batch, seq_len, d_model]
        mask:       [batch, seq_len] — attention mask (1 = реальный токен, 0 = паддинг)
        moe_out:    [batch, seq_len, d_model]
        aux_loss:   скаляр
        выход:      (x [batch, seq_len, d_model], aux_loss [скаляр])
"""

import torch
from torch import nn

from config import ModelConfig
from rmsnorm import RMSNorm
from attention import ROPE_MHSA
from need_to_do.moe_layer import MoELayer


class TransformerBlock(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # TODO: Создайте первую RMSNorm (перед Attention)
        # Подсказка: RMSNorm(model_cfg)
        self.ln_1 = ...  # ваш код здесь
        
        # TODO: Создайте Attention с RoPE
        # Подсказка: ROPE_MHSA(model_cfg)
        self.attention = ...  # ваш код здесь
        
        # TODO: Создайте вторую RMSNorm (перед MoE)
        self.ln_2 = ...  # ваш код здесь
        
        # TODO: Создайте MoE слой
        # Подсказка: MoELayer(model_cfg)
        self.moe = ...  # ваш код здесь
        
    def forward(self, x, mask):
        """
        Аргументы:
            x:    [batch, seq_len, d_model] — скрытое состояние
            mask: [batch, seq_len] — attention mask (1 = токен, 0 = паддинг)
            
        Возвращает:
            (x, aux_loss) — обновлённое скрытое состояние и балансировочный лосс
        """
        
        # TODO: Шаг 1 — Attention с residual connection
        # Порядок: ln_1(x) → attention(..., mask) → x + результат
        x = x + ...  # ваш код здесь
        
        # TODO: Шаг 2 — MoE слой
        # Подсказка: self.moe возвращает (output, aux_loss)
        # Порядок: ln_2(x) → moe(...) → получаем (moe_out, aux_loss)
        moe_out, aux_loss = ...  # ваш код здесь
        
        # TODO: Шаг 3 — Residual connection для MoE
        x = x + ...  # ваш код здесь
        
        # TODO: Шаг 4 — Верните кортеж
        return ...  # ваш код здесь