# MoE Transformer

Реализация Mixture of Experts трансформера на PyTorch.

## Структура

```
├── attention.py          # готово: Multi-Head Attention с RoPE + causal mask
├── rmsnorm.py            # готово: RMS Normalization
├── rope.py               # готово: Rotary Positional Embeddings
├── config.py             # готово: ModelConfig, TrainConfig
├── data.py               # готово: загрузка Шекспира, токенизация, DataLoader
├── trainer.py            # готово: цикл обучения, валидация, чекпоинты
├── training.py           # готово: точка входа
│
├── need_to_do/           # задания (дописать код)
│   ├── expert.py
│   ├── router.py
│   ├── moe_layer.py
│   ├── transformer_block.py
│   └── transformer.py
│
└── examples/             # готовые решения для проверки
```

## Порядок выполнения

1. **expert.py** — класс `ExpertMLP`: `Linear → GELU → Linear`
2. **router.py** — класс `Router` (gate + top-k) и `calc_load_balancing_loss`
3. **moe_layer.py** — класс `MoELayer`: роутер + ModuleList экспертов, цикл по экспертам, взвешенная сумма
4. **transformer_block.py** — класс `TransformerBlock`: `RMSNorm → Attention → RMSNorm → MoE`, residual connections
5. **transformer.py** — класс `MoETransformer`: Embedding, ModuleList блоков, финальная норма, unembedding

## Зависимости

```
config.py
  └── expert.py
        └── router.py
              └── moe_layer.py (зависит от expert + router)
                    └── transformer_block.py (зависит от moe_layer + attention + rmsnorm)
                          └── transformer.py (зависит от transformer_block + rmsnorm)
```

## Запуск

```bash
pip install torch transformers tqdm
python -m training
```

Модель скачает Шекспира, обучится за 3 эпохи и выведет метрики.