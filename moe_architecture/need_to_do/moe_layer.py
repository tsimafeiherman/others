"""
    Класс MoELayer (Mixture of Experts Layer).

    Это центральный компонент MoE-трансформера. Вместо одного MLP здесь работает
    целый набор экспертов, и роутер выбирает top-k из них для каждого токена.

    В __init__:
        - Создаёт экземпляр Router (выбирает экспертов для токенов)
        - Создаёт nn.ModuleList из ExpertMLP (число = n_experts)

    В forward(x):
        1. Получить top_k_probs, top_k_indices, router_logits от роутера.
        2. Инициализировать output = zeros_like(x).
        3. Для каждого эксперта i (цикл по всем экспертам):
           a. Найти токены, которым назначен этот эксперт:
              expert_mask = (top_k_indices == i).any(dim=-1) → [batch, seq_len] bool.
           b. Если такие токены есть (expert_mask.any()):
              - Извлечь их: expert_input = x[expert_mask] (форма [N, d_model])
              - Прогнать через эксперта: expert_output = self.experts[i](expert_input)
              - Вычислить вес эксперта для каждого токена:
                * token_pos_mask = (top_k_indices == i) — где эксперт в топ-k
                * Умножить на top_k_probs (вероятности)
                * Суммировать по последнему измерению (dim=-1)
                * Применить expert_mask и добавить размерность (unsqueeze(-1))
              - Добавить взвешенный выход: output[expert_mask] += expert_out * weights
        4. Вычислить aux_loss:
           - Если self.training: вызвать self.router.calc_load_balancing_loss(...)
           - Иначе: aux_loss = torch.tensor(0.0)
        5. Вернуть (output, aux_loss)

    Важно:
        - Сумма весов для одного токена = 1 (благодаря перенормализации в роутере),
          поэтому += даст правильную взвешенную сумму.
        - Всегда возвращайте кортеж (output, aux_loss), даже если aux_loss=0.
        - Цикл идёт ПО ЭКСПЕРТАМ (не по токенам) — это эффективнее.

    Размерности:
        x:              [batch, seq_len, d_model]
        top_k_probs:    [batch, seq_len, k_experts]
        top_k_indices:  [batch, seq_len, k_experts] (тип long)
        logits:         [batch, seq_len, n_experts]
        expert_mask:    [batch, seq_len] (bool)
        expert_input:   [N, d_model] (N — сколько токенов у этого эксперта)
        expert_output:  [N, d_model]
        expert_weights: [N, 1]
        output:         [batch, seq_len, d_model]
        aux_loss:       скаляр (0.0 или тензор)
"""

import torch
from torch import nn

from config import ModelConfig
from need_to_do.expert import ExpertMLP
from need_to_do.router import Router

class MoELayer(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_cfg = model_cfg
        
        # TODO: Создайте nn.ModuleList из ExpertMLP
        # Подсказка: [ExpertMLP(model_cfg) for _ in range(model_cfg.n_experts)]
        self.experts = ...  # ваш код здесь
        
        # TODO: Создайте экземпляр Router
        self.router = ...  # ваш код здесь
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        
        # TODO: Шаг 1 — получите top_k_probs, top_k_indices, logits от роутера
        # Подсказка: router возвращает три тензора
        top_k_probs, top_k_indices, logits = ...  # ваш код здесь
        
        # TODO: Шаг 2 — инициализируйте выходной тензор нулями
        output = ...  # ваш код здесь
        
        # TODO: Шаг 3 — цикл по всем экспертам
        for expert_idx in range(self.model_cfg.n_experts):
            
            # TODO: Шаг 3a — найдите токены, которым назначен этот эксперт
            # Подсказка: (top_k_indices == expert_idx).any(dim=-1) → [batch, seq_len] bool
            expert_mask = ...  # ваш код здесь

            # TODO: Шаг 3b — если есть хотя бы один токен для этого эксперта
            if ...:  # проверьте, есть ли True в expert_mask
                
                # TODO: Извлеките токены для этого эксперта
                # Подсказка: используйте expert_mask для индексации x
                expert_input = ...  # ваш код здесь — форма [N, d_model]
                
                # TODO: Прогоните через эксперта
                expert_output = ...  # ваш код здесь — форма [N, d_model]
                
                # TODO: Шаг 3c — вычислите веса эксперта для каждого токена
                # 1. Найдите позиции эксперта в top-k:
                #    token_pos_mask = (top_k_indices == expert_idx) — [batch, seq_len, k_experts] bool
                token_pos_mask = ...  # ваш код здесь
                
                # 2. Умножьте на вероятности и преобразуйте во float:
                #    expert_weights = token_pos_mask.float() * top_k_probs
                #    → [batch, seq_len, k_experts]
                expert_weights = ...  # ваш код здесь
                
                # 3. Суммируйте по последнему измерению (dim=-1):
                #    → [batch, seq_len] — один вес на токен
                expert_weights = ...  # ваш код здесь
                
                # 4. Оставьте только токены, которые идут к этому эксперту:
                #    → [N]
                filtered_weights = ...  # ваш код здесь
                
                # 5. Добавьте размерность для умножения:
                #    .unsqueeze(-1) → [N, 1]
                filtered_weights = ...  # ваш код здесь
                
                # TODO: Шаг 3d — добавьте взвешенный выход эксперта в output
                # Подсказка: output[expert_mask] += expert_output * filtered_weights
                output[expert_mask] += ...  # ваш код здесь
        
        # TODO: Шаг 4 — вычислите auxiliary loss
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            # Подсказка: используйте self.router.calc_load_balancing_loss(top_k_indices, logits)
            aux_loss = ...  # ваш код здесь
        
        # TODO: Шаг 5 — верните кортеж (output, aux_loss)
        return ...  # ваш код здесь