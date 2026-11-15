import matplotlib.pyplot as plt

def plot_loss(loss_history: list):
    """Строит график ошибки."""
    if not loss_history:
        print("История ошибок пуста. Сначала обучите сеть.")
        return
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title('История ошибки (MSE)')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.grid(True)
    plt.show()