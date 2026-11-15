import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из файла .env, если он есть
load_dotenv()

class Config:
    """Класс для хранения конфигурации приложения."""
    def __init__(self):
        # Значения по умолчанию
        self.default_data_dir = os.getenv("DATA_DIR", "data/input")
        self.default_output_dir = os.getenv("OUTPUT_DIR", "data/output")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        
        # Преобразуем строковые пути в объекты Path (pathlib)
        self.output_dir = Path(self.default_output_dir)
        
        # Создаем выходную директорию, если её нет
        self.output_dir.mkdir(parents=True, exist_ok=True)