import argparse
import subprocess
import sys
import logging
from pathlib import Path

from app.config import Config
from app.scanner import DataScanner
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)

def run_prepare(args, config):
    """Логика команды prepare."""
    scanner = DataScanner(allowed_extensions=args.ext.split(','))
    
    # Если путь не указан, берем из конфига
    input_path = args.path if args.path else config.default_data_dir
    
    # Формируем путь для сохранения
    output_path = Path(args.output) if args.output else config.output_dir / "dataset.npy"
    
    logger.info(f"Запуск подготовки данных из {input_path}")
    files = scanner.scan_directory(input_path)
    
    if files:
        scanner.process_files(files, str(output_path))
    else:
        logger.warning("Файлы с указанными расширениями не найдены.")

def run_doctor():
    """Логика команды doctor. Проверяет окружение."""
    print("--- Диагностика окружения ---")
    
    # 1. Версия Python
    print(f"Версия Python: {sys.version.split()[0]}")
    
    # 2. Наличие библиотек
    try:
        import numpy
        print(f"NumPy: установлен (версия {numpy.__version__})")
    except ImportError:
        print("NumPy: НЕ УСТАНОВЛЕН")
        
    try:
        import PIL
        print(f"Pillow (PIL): установлен")
    except ImportError:
        print("Pillow (PIL): НЕ УСТАНОВЛЕН")
        
    try:
        import tensorflow as tf
        print(f"TensorFlow: установлен (версия {tf.__version__})")
    except ImportError:
        print("TensorFlow: не установлен (это нормально, если используется PyTorch или NumPy)")
        
    try:
        import torch
        print(f"PyTorch: установлен (версия {torch.__version__})")
        # Проверка CUDA в PyTorch
        if torch.cuda.is_available():
            print(f"  CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA недоступна.")
    except ImportError:
        print("PyTorch: не установлен.")

    # 3. Проверка CUDA через nvidia-smi (subprocess)
    print("Проверка CUDA через nvidia-smi...")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            # Берем только первую строку с названием видеокарты
            gpu_info = result.stdout.split('\n')[8].strip() 
            print(f"GPU найдена: {gpu_info}")
        else:
            print("nvidia-smi вернул ошибку. CUDA может быть не установлена.")
    except FileNotFoundError:
        print("Утилита nvidia-smi не найдена (CUDA не установлена или нет GPU Nvidia).")

    # 4. Текущая рабочая директория
    print(f"Текущая рабочая директория: {Path.cwd()}")

def main():
    config = Config()
    setup_logging(level=config.log_level)
    
    parser = argparse.ArgumentParser(description="CLI-утилита для подготовки данных и диагностики.")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")
    
    # Команда prepare
    parser_prepare = subparsers.add_parser("prepare", help="Подготовка данных из файлов в .npy")
    parser_prepare.add_argument("--path", type=str, help="Путь к директории с данными")
    parser_prepare.add_argument("--ext", type=str, required=True, help="Расширения файлов через запятую (например, jpg,png,csv)")
    parser_prepare.add_argument("--output", type=str, help="Путь для сохранения .npy файла")
    
    # Команда doctor
    parser_doctor = subparsers.add_parser("doctor", help="Проверка окружения")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        run_prepare(args, config)
    elif args.command == "doctor":
        run_doctor()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()