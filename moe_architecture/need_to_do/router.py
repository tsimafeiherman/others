"""
    Класс Router (роутер / gate).

    Выбирает, какие эксперты будут обрабатывать каждый токен.

    Ключевые моменты:
    - gate = Linear(d_model, n_experts, bias=False)
      (bias=False, чтобы роутер не мог "запомнить" любимого эксперта)
    - Инициализация с малой дисперсией (0.01)
      (чтобы в начале все эксперты получали примерно равные вероятности)
    - Перенормализация top-k вероятностей
      (сумма весов выбранных экспертов должна быть = 1)

    forward(x):
        1. logits = gate(x) — сырые оценки для каждого эксперта
        2. probs = softmax(logits, dim=-1) — вероятности
        3. top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1) — выбор лучших
        4. Перенормализация: top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        5. Возврат: top_k_probs, top_k_indices, router_logits (логиты нужны для балансировки)

    Функция calc_load_balancing_loss(router_logits, top_k_indices):
        Штрафует модель, если эксперты используются неравномерно.

        Шаги:
        1. Превратить логиты и индексы в плоские [tokens, ...] (batch*seq_len)
        2. Создать маску использования экспертов:
           - F.one_hot(top_k_indices, n_experts) → [tokens, k, n_experts]
           - .sum(dim=1) → [tokens, n_experts]
           - (mask > 0).float() — бинаризация (на случай двойного выбора)
        3. fraction = mask.mean(dim=0) — реальная доля токенов у каждого эксперта [n_experts]
        4. gates_mean = softmax(router_logits).mean(dim=0) — средняя "желаемая" вероятность [n_experts]
        5. aux_loss = sum(fraction * gates_mean) * n_experts
           (при идеальном балансе fraction = gates_mean = 1/n_experts, aux_loss = 1)

    Размерности:
        x:              [batch, seq_len, d_model]
        logits:         [batch, seq_len, n_experts]
        probs:          [batch, seq_len, n_experts]
        top_k_probs:    [batch, seq_len, k_experts]
        top_k_indices:  [batch, seq_len, k_experts] (тип long)
        mask:           [tokens, n_experts]
        fraction:       [n_experts]
        gates_mean:     [n_experts]
        aux_loss:       скаляр
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class Router(nn.Module):
    def __init__(self, model_cfg: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model_cfg = model_cfg
        
        # TODO: Создайте линейный слой d_model → n_experts, БЕЗ bias
        # Подсказка: nn.Linear(model_cfg.d_model, model_cfg.n_experts, bias=False)
        self.gate = ...  # ваш код здесь
        
        # TODO: Инициализируйте веса нормальным распределением с малой дисперсией (0.01)
        # Подсказка: nn.init.normal_(self.gate.weight, std=0.01)
        ...  # ваш код здесь
    
    def calc_load_balancing_loss(self, top_k_indices, router_logits):
        """
        Вычисляет вспомогательный лосс для равномерной загрузки экспертов.
        
        Аргументы:
            top_k_indices: [batch, seq_len, k_experts] — индексы выбранных экспертов
            router_logits:  [batch, seq_len, n_experts] — сырые логиты роутера (ДО softmax!)
            
        Возвращает:
            aux_loss: скаляр
        """
        
        # TODO: Шаг 1 — превратите в плоские тензоры [batch*seq_len, ...]
        # Подсказка: .view(-1, ...)
        top_k_indices = ...  # ваш код здесь — [tokens, k_experts]
        router_logits = ...  # ваш код здесь — [tokens, n_experts]
        
        # TODO: Шаг 2 — создайте маску использования экспертов
        # a) One-hot кодирование: F.one_hot(top_k_indices, num_classes=n_experts)
        #    → [tokens, k_experts, n_experts]
        mask = ...  # ваш код здесь
        
        # b) Суммируйте по измерению k_experts (dim=1)
        #    → [tokens, n_experts]
        mask = ...  # ваш код здесь
        
        # c) Бинаризуйте: (mask > 0).float()
        #    → [tokens, n_experts] с 0 и 1
        mask = ...  # ваш код здесь
        
        # TODO: Шаг 3 — реальная доля токенов у каждого эксперта
        # Подсказка: mask.mean(dim=0) → [n_experts]
        fraction = ...  # ваш код здесь
        
        # TODO: Шаг 4 — средняя "желаемая" вероятность эксперта
        # a) Softmax от логитов (по последней размерности)
        router_probs = ...  # ваш код здесь
        
        # b) Среднее по токенам (dim=0)
        gates_mean = ...  # ваш код здесь
        
        # TODO: Шаг 5 — auxiliary loss
        # Подсказка: (fraction * gates_mean).sum() * n_experts
        aux_loss = ...  # ваш код здесь
        
        return aux_loss
    
    def forward(self, x):
        """
        Аргументы:
            x: [batch, seq_len, d_model] — скрытое состояние после Attention + RMSNorm
            
        Возвращает:
            top_k_probs:   [batch, seq_len, k_experts] — перенормализованные веса
            top_k_indices: [batch, seq_len, k_experts] — индексы выбранных экспертов
            logits:        [batch, seq_len, n_experts] — сырые логиты (для aux loss)
        """
        
        # TODO: Шаг 1 — сырые оценки для каждого эксперта
        # Подсказка: self.gate(x) → [batch, seq_len, n_experts]
        logits = ...  # ваш код здесь
        
        # TODO: Шаг 2 — вероятности через softmax по последней размерности
        probs = ...  # ваш код здесь
        
        # TODO: Шаг 3 — выберите top-k экспертов
        # Подсказка: torch.topk(probs, k=self.model_cfg.k_experts, dim=-1)
        # Возвращает (values, indices) — обе [batch, seq_len, k_experts]
        top_k_probs, top_k_indices = ...  # ваш код здесь
        
        # TODO: Шаг 4 — перенормализуйте вероятности (сумма = 1)
        # Подсказка: top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        top_k_probs = ...  # ваш код здесь
        
        return top_k_probs, top_k_indices, logits