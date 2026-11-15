import logging
import sys

def setup_logging(level=logging.INFO):
    """Настраивает базовое логирование для вывода в консоль."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )