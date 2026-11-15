import os
from app.producer_consumer import ProducerConsumerProcessor
from app.performance import save_performance_report

def main():
    # Укажите путь к папке с картинками
    INPUT_DIR = "data/input" 
    
    # Убедимся, что папка существует
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Создана папка {INPUT_DIR}. Положите туда картинки (jpg, png) и запустите снова.")
        return

    # Инициализируем процессор
    processor = ProducerConsumerProcessor(
        input_dir=INPUT_DIR, 
        num_threads=3,     # Количество потоков для чтения
        num_processes=4    # Количество процессов для аугментации
    )
    
    # 1. Запускаем последовательную версию (для сравнения)
    seq_time = processor.run_sequential()
    
    # 2. Запускаем параллельную версию
    par_time = processor.run_parallel()
    
    # 3. Сохраняем отчет
    save_performance_report(
        seq_time=seq_time, 
        par_time=par_time, 
        files_count=processor.total_files, 
        output_file="performance_report.json"
    )

if __name__ == "__main__":
    # Важно для Windows: защита точки входа для multiprocessing
    main()