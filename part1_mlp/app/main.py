import os
import numpy as np
from app.models import NeuralNetwork
from app.manager import DatasetManager
from app.utils import plot_loss
from app.exceptions import NeuralNetworkError

def main():
    net = None
    dataset = DatasetManager()
    
    while True:
        print("\n--- Меню ---")
        print("1. Создать сеть")
        print("2. Загрузить данные из CSV")
        print("3. Обучить сеть")
        print("4. Сделать предсказание")
        print("5. Сохранить веса")
        print("6. Загрузить веса")
        print("7. Показать график ошибки")
        print("0. Выход")
        
        choice = input("Выберите действие: ").strip()
        
        try:
            if choice == '1':
                in_size = int(input("Введите размер входного слоя: "))
                hid_size = int(input("Введите размер скрытого слоя: "))
                out_size = int(input("Введите размер выходного слоя: "))
                act = input("Функция активации (sigmoid/relu): ").strip()
                net = NeuralNetwork(in_size, hid_size, out_size, act)
                print("Сеть создана.")
                
            elif choice == '2':
                path = input("Введите путь к CSV файлу: ").strip()
                dataset.load_csv(path)
                dataset.split_data()
                
            elif choice == '3':
                if net is None or dataset.X_train is None:
                    print("Сначала создайте сеть и загрузите данные!")
                    continue
                lr = float(input("Скорость обучения (например, 0.1): "))
                epochs = int(input("Количество эпох: "))
                batch = int(input("Размер батча: "))
                print("Начинаем обучение...")
                net.train(dataset.X_train, dataset.y_train, epochs, lr, batch)
                
            elif choice == '4':
                if net is None:
                    print("Сначала создайте сеть!")
                    continue
                
                print("Введите признаки через пробел (или нажмите Enter для теста на случайном примере):")
                user_input = input().strip()
                if user_input:
                    try:
                        X_new = np.array([float(x) for x in user_input.split()]).reshape(1, -1)
                    except ValueError:
                        print("Ошибка ввода. Нужны числа.")
                        continue
                else:
                    if dataset.X_test is None:
                        print("Нет тестовых данных.")
                        continue
                    X_new = dataset.X_test[:1]
                    
                pred = net.predict(X_new)
                print(f"Предсказание: {pred.flatten()}")
                
            elif choice == '5':
                if net is None: continue
                path = input("Введите имя файла для сохранения (например, weights.json): ")
                net.save_weights(path)
                
            elif choice == '6':
                if net is None:
                    print("Сначала создайте сеть (структура должна совпадать).")
                    continue
                path = input("Введите имя файла для загрузки: ")
                net.load_weights(path)
                
            elif choice == '7':
                if net is None: continue
                plot_loss(net.loss_history)
                
            elif choice == '0':
                print("Выход.")
                break
            else:
                print("Неверный ввод.")
                
        except NeuralNetworkError as e:
            print(f"Ошибка нейросети: {e}")
        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")

if __name__ == "__main__":
    # Для корректного импорта, если запускать из папки part1_mlp
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()