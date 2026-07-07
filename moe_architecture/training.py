"""
    training.py — точка входа для обучения MoE-трансформера.

    Запускать в консоли из корня MoE architecture используя python -m training

    Что делает:
    1. Загружает конфиги модели и обучения.
    2. Загружает токенизатор GPT-2.
    3. Создаёт даталоадеры для обучения и валидации.
    4. Создаёт Trainer и запускает обучение.
    5. Сохраняет чекпоинт после каждой эпохи.
"""

import torch
from transformers import AutoTokenizer

from config import ModelConfig, TrainConfig
from data import get_dataloaders
from trainer import Trainer


def main():
    print("=" * 60)
    print("Запуск обучения MoE-трансформера")
    print("=" * 60)

    # 1. Конфигурации
    model_cfg = ModelConfig(
        d_model=256,        # уменьшим для быстрого теста
        layers=4,           # 4 слоя вместо 12
        n_heads=8,
        head_dim=32,
        mlp_mult=4,
        n_ctx=512,
        vocab_size=50257,   # размер словаря GPT-2
        n_experts=8,
        k_experts=2,
        load_balancing_weight=0.01,
    )

    train_cfg = TrainConfig(
        epochs=3,           # для теста 3 эпохи
        save_model_dir="checkpoints",
        save_logs_dir="logs",
        lr=3e-4,
    )

    print(f"\nКонфигурация модели:")
    print(f"  d_model={model_cfg.d_model}, layers={model_cfg.layers}")
    print(f"  n_heads={model_cfg.n_heads}, head_dim={model_cfg.head_dim}")
    print(f"  n_experts={model_cfg.n_experts}, k_experts={model_cfg.k_experts}")
    print(f"  vocab_size={model_cfg.vocab_size}")
    print(f"\nКонфигурация обучения:")
    print(f"  epochs={train_cfg.epochs}, lr={train_cfg.lr}")
    print(f"  save_dir={train_cfg.save_model_dir}")

    # 2. Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nУстройство: {device}")

    # 3. Загружаем токенизатор
    print("\nЗагрузка токенизатора GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    # 4. Создаём даталоадеры
    print("Подготовка данных...")
    train_loader, val_loader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=8,
        val_split=0.05,
        min_len=200,
        max_len=300,
        seed=42,
        max_train_samples=20, 
    )

    # 5. Проверяем один батч
    print("\nПроверка даталоадера:")
    for batch in train_loader:
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  mask shape: {mask.shape}")
        print(f"  Пример (первые 30 токенов): {tokenizer.decode(input_ids[0, :30])}...")
        break

    # 6. Создаём Trainer
    print("\nСоздание модели и trainer'а...")
    trainer = Trainer(train_cfg, model_cfg, train_loader, val_loader)

    # 7. Выводим количество параметров
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\nПараметры модели:")
    print(f"  Всего: {total_params:,}")
    print(f"  Обучаемых: {trainable_params:,}")

    # 8. Запускаем обучение
    print("\n" + "=" * 60)
    print("Начало обучения")
    print("=" * 60 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем.")
        print("Сохраняем последний чекпоинт...")
        trainer.save(epoch=train_cfg.epochs, file_name="interrupted.pth")
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        raise

    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print("=" * 60)


if __name__ == "__main__":
    main()