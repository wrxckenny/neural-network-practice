import os
import time
import queue
import threading
import multiprocessing
from pathlib import Path
from PIL import Image
import numpy as np

from app.augment import apply_augmentation

class ProducerConsumerProcessor:
    def __init__(self, input_dir: str, num_threads: int = 2, num_processes: int = 2):
        self.input_dir = Path(input_dir)
        self.num_threads = num_threads
        self.num_processes = num_processes
        
        # Очереди для обмена данными
        self.file_path_queue = queue.Queue()   # Пути к файлам
        self.image_data_queue = queue.Queue()  # Загруженные массивы
        
        # Счетчик и блокировка для потокобезопасности
        self.processed_count = 0
        self.counter_lock = threading.Lock()
        
        self.total_files = 0

    def producer(self):
        """Поток-производитель: сканирует папку и кладет пути в очередь."""
        extensions = ['.jpg', '.jpeg', '.png']
        files = []
        for ext in extensions:
            files.extend(self.input_dir.rglob(f"*{ext}"))
        
        self.total_files = len(files)
        print(f"[Производитель] Найдено файлов: {self.total_files}")

        for file_path in files:
            self.file_path_queue.put(file_path)
        
        # Сигналы завершения для потребителей (по одному на каждый поток)
        for _ in range(self.num_threads):
            self.file_path_queue.put(None)

    def consumer(self):
        """Поток-потребитель: читает пути из очереди, загружает картинки и кладет в другую очередь."""
        while True:
            file_path = self.file_path_queue.get()
            if file_path is None:
                # Сигнал о завершении работы потоков
                self.file_path_queue.task_done()
                break
            
            try:
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img)
                self.image_data_queue.put(img_array)
            except Exception as e:
                print(f"[Потребитель] Ошибка чтения {file_path}: {e}")
            finally:
                self.file_path_queue.task_done()

    def run_parallel(self):
        """Запускает параллельную обработку с использованием потоков и процессов."""
        print("\n--- Запуск параллельной обработки (Потоки + Процессы) ---")
        start_time = time.time()
        
        # 1. Запускаем поток-производитель
        producer_thread = threading.Thread(target=self.producer)
        producer_thread.start()
        
        # 2. Запускаем потоки-потребители
        consumer_threads = []
        for _ in range(self.num_threads):
            t = threading.Thread(target=self.consumer)
            t.start()
            consumer_threads.append(t)
            
        # 3. Запускаем пул процессов для аугментации
        augmented_results = []
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # Ждем, пока потоки загрузят все данные в очередь
            producer_thread.join()
            for t in consumer_threads:
                t.join()
            
            # Теперь очередь image_data_queue заполнена.
            while not self.image_data_queue.empty():
                img_array = self.image_data_queue.get()
                
                # Отправляем задачу в пул процессов
                result = pool.apply_async(process_worker, (img_array,))
                augmented_results.append(result)
            
            # Собираем результаты
            for res in augmented_results:
                res.get()
            
            # Обновляем счетчик
            with self.counter_lock:
                self.processed_count += len(augmented_results)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[Параллельно] Обработано файлов: {self.processed_count}, Время: {elapsed:.2f} сек.")
        return elapsed

    def run_sequential(self):
        """Последовательная обработка для сравнения."""
        print("\n--- Запуск последовательной обработки ---")
        start_time = time.time()
        
        extensions = ['.jpg', '.jpeg', '.png']
        files = []
        for ext in extensions:
            files.extend(self.input_dir.rglob(f"*{ext}"))
            
        count = 0
        for file_path in files:
            try:
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img)
                apply_augmentation(img_array)
                count += 1
            except Exception as e:
                print(f"[Последовательно] Ошибка: {e}")
                
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"[Последовательно] Обработано файлов: {count}, Время: {elapsed:.2f} сек.")
        return elapsed


# --- ФУНКЦИЯ ВНЕ КЛАССА (для multiprocessing) ---
def process_worker(img_array):
    """Функция для пула процессов."""
    from app.augment import apply_augmentation
    return apply_augmentation(img_array)