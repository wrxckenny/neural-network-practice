import json
import time

def save_performance_report(seq_time: float, par_time: float, files_count: int, output_file: str = "performance_report.json"):
    """Сохраняет результаты замеров в JSON-файл."""
    report = {
        "total_files_processed": files_count,
        "sequential_time_seconds": round(seq_time, 4),
        "parallel_time_seconds": round(par_time, 4),
        "speedup": round(seq_time / par_time, 2) if par_time > 0 else 0
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"\nОтчет сохранен в {output_file}")
    print(f"Последовательное время: {report['sequential_time_seconds']} сек.")
    print(f"Параллельное время: {report['parallel_time_seconds']} сек.")
    print(f"Ускорение: {report['speedup']}x")