class NeuralNetworkError(Exception):
    """Базовое исключение для всех ошибок нейросети."""
    pass

class InvalidLayerSizeError(NeuralNetworkError):
    """Вызывается, если размеры слоев некорректны."""
    pass

class MismatchedDataError(NeuralNetworkError):
    """Вызывается при несовместимости размерностей данных и сети."""
    pass

class FileFormatError(NeuralNetworkError):
    """Вызывается при ошибке чтения файла."""
    pass