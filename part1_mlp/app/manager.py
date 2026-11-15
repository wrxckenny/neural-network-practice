import numpy as np
import pandas as pd
from .exceptions import FileFormatError, MismatchedDataError

class DatasetManager:
    """
    Класс для загрузки, нормализации и разбиения данных.
    """
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_params = {}

    def load_csv(self, filepath: str):
        """Загружает данные из CSV. Последний столбец - целевая переменная."""
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise FileFormatError(f"Ошибка чтения файла: {e}")

        if df.empty:
            raise FileFormatError("Файл пуст.")

        # Предполагаем, что последний столбец - это y
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.reshape(-1, 1)
        
        if X.shape[1] == 0:
            raise MismatchedDataError("В данных нет признаков (только целевая переменная).")

        # Нормализация (Min-Max)
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        # Избегаем деления на ноль
        range_val = self.X_max - self.X_min
        range_val[range_val == 0] = 1 
        
        X_norm = (X - self.X_min) / range_val
        
        # Нормализация y (если это регрессия, но для простоты оставим как есть или тоже min-max)
        # Для классификации y должен быть 0/1, здесь просто оставим без изменений, 
        # так как сигмоида выдает 0..1
        self.X_data = X_norm
        self.y_data = y
        
        print(f"Данные загружены. Признаков: {X.shape[1]}, Примеров: {X.shape[0]}")

    def split_data(self, test_size: float = 0.2):
        """Разбивает данные на train/test."""
        if self.X_data is None:
            raise ValueError("Сначала загрузите данные.")
            
        num_samples = self.X_data.shape[0]
        indices = np.random.permutation(num_samples)
        split_idx = int(num_samples * (1 - test_size))
        
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        self.X_train = self.X_data[train_idx]
        self.X_test = self.X_data[test_idx]
        self.y_train = self.y_data[train_idx]
        self.y_test = self.y_data[test_idx]
        
        print(f"Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")