"""
    Класс MoETransformer — полная модель Mixture of Experts Transformer.

    Это декадер-only трансформер (как GPT) с MoE слоями вместо обычных MLP.

    Архитектура (Pre-LN стиль):
        1. Embedding:         input_ids → векторы токенов [batch, seq_len, d_model]
        2. TransformerBlocks: × n_layers (каждый: Attention → MoE, с residual)
        3. Final RMSNorm:     нормализация перед выходом
        4. Unembedding:       векторы → логиты [batch, seq_len, vocab_size]

    __init__(cfg):
        - embed:     nn.Embedding(vocab_size, d_model) — эмбеддинги токенов
        - layers:    nn.ModuleList([TransformerBlock(cfg) × n_layers]) — блоки
        - ln_final:  RMSNorm(cfg) — финальная нормализация
        - final_proj: nn.Linear(d_model, vocab_size) — выходная проекция (unembedding)

    forward(input_ids, mask):
        1. x = self.embed(input_ids) — токены → эмбеддинги
        2. Для каждого блока в self.layers:
           x, aux = block(x, mask)
           total_aux += aux — суммируем балансировочный лосс со всех слоёв
        3. x = self.ln_final(x) — финальная нормализация
        4. logits = self.final_proj(x) — в логиты
        5. return logits, total_aux

    Примечания:
        - Позиционные эмбеддинги уже внутри Attention (через RoPE),
          поэтому здесь их не добавляем.
        - total_aux — сумма aux_loss со всех MoE-слоёв.
          При обучении добавляется к основному лоссу с весом load_balancing_weight.
        - Размер vocab_size должен совпадать с токенизатором (для GPT-2 это 50257).

    Размерности:
        input_ids:   [batch, seq_len] — индексы токенов
        mask:        [batch, seq_len] — attention mask (1 = токен, 0 = паддинг)
        x:           [batch, seq_len, d_model] — скрытое состояние
        logits:      [batch, seq_len, vocab_size] — предсказания
        total_aux:   скаляр — суммарный балансировочный лосс
"""

import torch
from torch import nn

from config import ModelConfig
from need_to_do.transformer_block import TransformerBlock
from rmsnorm import RMSNorm


class MoETransformer(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # TODO: Создайте эмбеддинги токенов
        # Подсказка: nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.embed = ...  # ваш код здесь
        
        # TODO: Создайте список блоков трансформера
        # Подсказка: nn.ModuleList([TransformerBlock(model_cfg) for _ in range(model_cfg.layers)])
        self.layers = ...  # ваш код здесь
        
        # TODO: Создайте финальную RMSNorm
        # Подсказка: RMSNorm(model_cfg)
        self.ln_final = ...  # ваш код здесь
        
        # TODO: Создайте выходную проекцию (unembedding)
        # Подсказка: nn.Linear(d_model, vocab_size)
        # Превращает скрытое состояние в логиты для каждого токена словаря
        self.final_proj = ...  # ваш код здесь
        
    def forward(self, input_ids, mask):
        """
        Аргументы:
            input_ids: [batch, seq_len] — индексы токенов
            mask:      [batch, seq_len] — attention mask (1 = токен, 0 = паддинг)
            
        Возвращает:
            logits:    [batch, seq_len, vocab_size] — логиты для следующего токена
            total_aux: скаляр — суммарный балансировочный лосс со всех MoE-слоёв
        """
        
        # TODO: Шаг 1 — эмбеддинги токенов
        # Подсказка: self.embed(input_ids) → [batch, seq_len, d_model]
        x = ...  # ваш код здесь
        
        # TODO: Шаг 2 — прогон через все блоки
        # Для каждого блока:
        #   x, aux = layer(x, mask)
        #   total_aux += aux
        total_aux = 0.0
        for layer in self.layers:
            ...  # ваш код здесь
            
        # TODO: Шаг 3 — финальная нормализация
        x = ...  # ваш код здесь
        
        # TODO: Шаг 4 — проекция в логиты
        logits = ...  # ваш код здесь
        
        # TODO: Шаг 5 — вернуть логиты и суммарный aux_loss
        return ...  # ваш код здесь