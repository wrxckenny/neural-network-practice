import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class DataScanner:
    """Класс для сканирования директорий и подготовки данных."""
    
    def __init__(self, allowed_extensions: list):
        self.allowed_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in allowed_extensions]

    def scan_directory(self, directory_path: str) -> list:
        """Рекурсивно обходит директорию и возвращает список путей к файлам."""
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Директория не найдена: {directory_path}")
        
        files = []
        for ext in self.allowed_extensions:
            files.extend(path.rglob(f"*{ext}"))
        
        logger.info(f"Найдено файлов: {len(files)} в директории {directory_path}")
        return files

    def process_files(self, file_paths: list, output_file: str):
        """Читает файлы, нормализует и сохраняет в единый .npy массив."""
        start_time = time.time()
        data_list = []
        
        for file_path in file_paths:
            try:
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Чтение изображения и нормализация (деление на 255)
                    img = Image.open(file_path).convert('RGB')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    data_list.append(img_array)
                    
                elif file_path.suffix.lower() == '.csv':
                    # Чтение CSV (предполагаем, что все колонки числовые)
                    df = pd.read_csv(file_path)
                    # Заполняем пропуски нулями и конвертируем в массив
                    array = df.fillna(0).values.astype(np.float32)
                    data_list.append(array)
                    
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {file_path}: {e}")
                continue
        
        if not data_list:
            logger.warning("Не удалось обработать ни один файл. Массив не создан.")
            return None

        # Объединяем все данные в один массив (если размеры совпадают)
        try:
            combined_array = np.stack(data_list) # Создает 3D или 2D массив
        except ValueError as e:
             logger.error(f"Несовместимые размерности файлов: {e}. Сохраняем как список объектов.")
             # Если размеры разные, сохраняем как object array
             combined_array = np.array(data_list, dtype=object)

        # Сохраняем в .npy
        np.save(output_file, combined_array)
        
        end_time = time.time()
        logger.info(f"Статистика:")
        logger.info(f"  - Обработано файлов: {len(data_list)}")
        logger.info(f"  - Форма итогового массива: {combined_array.shape}")
        logger.info(f"  - Размер файла: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
        logger.info(f"  - Время обработки: {end_time - start_time:.2f} сек.")
        
        return combined_array