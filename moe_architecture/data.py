"""
    data.py — загрузка и подготовка данных для обучения MoE-трансформера.

    Что здесь происходит:
    1. Скачиваем тексты Шекспира (или загружаем локальный файл).
    2. Токенизируем через GPT-2 токенизатор.
    3. Нарезаем на непересекающиеся куски случайной длины (200-300 токенов).
    4. Создаём collate_fn — дополняет последовательности pad-токенами до максимальной длины в батче.
    5. Возвращаем DataLoader'ы для обучения и валидации.
"""

import urllib.request
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Optional


def download_shakespeare(url: str = None, save_path: str = "input.txt") -> str:
    """
    Скачивает датасет с текстами Шекспира, если файла нет локально.
    Возвращает путь к файлу.
    """
    if url is None:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Файл успешно скачан и сохранён как {save_path}")
    except Exception as e:
        print(f"Не удалось скачать файл: {e}")
        print("Убедитесь, что файл существует локально или проверьте подключение к интернету.")
        raise

    return save_path


class ShakespeareDataset(Dataset):
    """
    Датасет, который:
    - Принимает текст и токенизатор.
    - Токенизирует весь текст.
    - Нарезает его на непересекающиеся сэмплы случайной длины (200-300 токенов).
    - Хранит сэмплы как списки индексов токенов.
    """

    def __init__(self, text: str, tokenizer: AutoTokenizer, min_len: int = 200, max_len: int = 300, seed: int = 42):
        self.tokenizer = tokenizer
        self.texts: List[List[int]] = []

        random.seed(seed)

        # Токенизируем весь текст в список индексов
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"Текст токенизирован. Всего токенов: {len(ids)}")

        # Нарезаем на куски случайной длины
        i = 0
        while i < len(ids):
            split_size = random.randint(min_len, max_len)
            chunk = ids[i:i + split_size]
            # Добавляем только если кусок не слишком короткий
            if len(chunk) >= min_len // 2:
                self.texts.append(chunk)
            i += split_size

        print(f"Создано {len(self.texts)} сэмплов.")

    def __getitem__(self, index: int) -> List[int]:
        return self.texts[index]

    def __len__(self) -> int:
        return len(self.texts)


def collate_fn(batch: List[List[int]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Собирает батч из списка сэмплов разной длины.

    Аргументы:
        batch: список списков индексов токенов (List[List[int]]).
        pad_token_id: ID токена, который используется для паддинга.

    Возвращает:
        Словарь с ключами:
            "input_ids": тензор [batch_size, max_seq_len] — токены с паддингами.
            "attention_mask": тензор [batch_size, max_seq_len] — маска (1 = реальный токен, 0 = паддинг).
            "labels": тензор [batch_size, max_seq_len] — то же, что и input_ids (для удобства).
    """
    max_len = max(len(item) for item in batch)

    padded_inputs = []
    attention_masks = []

    for item in batch:
        pad_needed = max_len - len(item)

        # Паддинг справа
        padded_tensor = torch.tensor(item, dtype=torch.long)
        padded_tensor = torch.nn.functional.pad(padded_tensor, (0, pad_needed), value=pad_token_id)
        padded_inputs.append(padded_tensor)

        # Маска: 1 для реальных токенов, 0 для паддингов
        mask = torch.nn.functional.pad(torch.ones(len(item), dtype=torch.long), (0, pad_needed), value=0)
        attention_masks.append(mask)

    return {
        "input_ids": torch.stack(padded_inputs),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(padded_inputs),  # для языкового моделирования labels = input_ids
    }


def get_dataloaders(
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    val_split: float = 0.05,
    min_len: int = 200,
    max_len: int = 300,
    seed: int = 42,
    file_path: str = "input.txt",
    max_train_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Основная функция: загружает данные, создаёт датасеты и возвращает DataLoader'ы.

    Аргументы:
        tokenizer: токенизатор (GPT-2).
        batch_size: размер батча.
        val_split: доля данных для валидации (0.05 = 5%).
        min_len, max_len: границы случайной длины сэмплов.
        seed: зерно для воспроизводимости.
        file_path: путь к файлу с текстом.
        max_train_samples: если указано, обрезает train датасет до этого числа семплов.

    Возвращает:
        train_loader, val_loader — DataLoader'ы для обучения и валидации.
    """

    # 1. Загружаем текст
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Файл {file_path} загружен. Длина текста: {len(text)} символов.")
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Пытаемся скачать...")
        download_shakespeare(save_path=file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    # 2. Устанавливаем pad_token (GPT-2 не имеет отдельного pad-токена)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Установлен pad_token = eos_token (ID: {tokenizer.pad_token_id})")

    # 3. Создаём полный датасет
    full_dataset = ShakespeareDataset(text, tokenizer, min_len=min_len, max_len=max_len, seed=seed)

    # 4. Разделяем на train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"Train сэмплов: {train_size}, Val сэмплов: {val_size}")

    # 5. Обрезаем train до max_train_samples (если указано)
    if max_train_samples is not None and max_train_samples < train_size:
        indices = torch.randperm(train_size)[:max_train_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())
        print(f"Train обрезан до {max_train_samples} семплов")

    # 6. Создаём DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        drop_last=False,
    )

    return train_loader, val_loader