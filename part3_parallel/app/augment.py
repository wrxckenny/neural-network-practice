import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random

def apply_augmentation(image_array: np.ndarray) -> list:
    """
    Применяет набор аугментаций к изображению.
    Возвращает список из 3-х новых массивов (оригинал + 2 измененных).
    """
    # Конвертируем массив обратно в изображение PIL для удобства работы
    # (если на входе был нормализованный массив, умножаем на 255)
    if image_array.max() <= 1.0:
        img_data = (image_array * 255).astype(np.uint8)
    else:
        img_data = image_array.astype(np.uint8)
        
    img = Image.fromarray(img_data)
    augmented_results = []

    # 1. Оригинал
    augmented_results.append(np.array(img))

    # 2. Случайный поворот
    angle = random.randint(-30, 30)
    rotated = img.rotate(angle)
    augmented_results.append(np.array(rotated))

    # 3. Случайное отражение (горизонтальное)
    if random.choice([True, False]):
        flipped = ImageOps.mirror(img)
    else:
        flipped = ImageOps.flip(img)
    augmented_results.append(np.array(flipped))

    # 4. Добавление шума
    noise = np.random.normal(0, 10, img_data.shape).astype(np.int16)
    noisy_data = np.clip(img_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented_results.append(np.array(noisy_data))

    return augmented_results