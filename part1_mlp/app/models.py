import numpy as np
import json
from .exceptions import InvalidLayerSizeError, MismatchedDataError

class NeuralNetwork:
    """
    Полносвязная нейронная сеть с одним скрытым слоем.
    Поддерживает функции активации: sigmoid, relu.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation: str = 'sigmoid'):
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise InvalidLayerSizeError("Размеры слоев должны быть положительными числами.")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        
        if self.activation_name not in ['sigmoid', 'relu']:
            raise ValueError("Доступные функции активации: 'sigmoid', 'relu'")

        # Инициализация весов малыми случайными числами
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
        self.loss_history = []

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500))) # clip для избежания overflow
        elif self.activation_name == 'relu':
            return np.maximum(0, z)

    def _activate_derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.activation_name == 'sigmoid':
            return a * (1 - a)
        elif self.activation_name == 'relu':
            return (z > 0).astype(float)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Прямой проход."""
        if X.shape[1] != self.input_size:
            raise MismatchedDataError(f"Ожидалось {self.input_size} признаков, получено {X.shape[1]}")
        
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._activate(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._activate(self.z2) # Для выходного слоя используем ту же активацию для простоты
        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float):
        """Обратный проход и обновление весов (MSE loss)."""
        m = X.shape[0]
        
        # Градиенты для выходного слоя
        dz2 = (self.a2 - y) * self._activate_derivative(self.z2, self.a2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Градиенты для скрытого слоя
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self._activate_derivative(self.z1, self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Обновление весов
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 32):
        """Обучение на батчах с сохранением истории ошибок."""
        self.loss_history = []
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Перемешивание данных
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                # MSE Loss
                loss = np.mean((y_pred - y_batch) ** 2) 
                epoch_loss += loss
                
                self.backward(X_batch, y_batch, learning_rate)
                
            avg_loss = epoch_loss / (num_samples / batch_size)
            self.loss_history.append(avg_loss)
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание."""
        return self.forward(X)

    def save_weights(self, filepath: str):
        """Сохранение весов в JSON."""
        data = {
            'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
            'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
            'activation': self.activation_name,
            'sizes': [self.input_size, self.hidden_size, self.output_size]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Веса сохранены в {filepath}")

    def load_weights(self, filepath: str):
        """Загрузка весов из JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.W1 = np.array(data['W1'])
        self.b1 = np.array(data['b1'])
        self.W2 = np.array(data['W2'])
        self.b2 = np.array(data['b2'])
        print(f"Веса загружены из {filepath}")