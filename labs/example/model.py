# Линейная регрессия, обучаемая методом градиентного спуска.
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LinearRegressionGD:
    """
    learning_rate: Скорость обучения (шаг градиентного спуска).
    n_iterations: Количество итераций (эпох) обучения.
    weights: Вектор весов модели (инициализируется при обучении).
    bias: Свободный член (смещение) модели.
    loss_history: История значений функции потерь (MSE) по эпохам.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
    ) -> None:
        """
        Инициализирует гиперпараметры модели.
        Args:
            learning_rate: Скорость обучения. Должна быть положительным числом.
                Слишком большое значение может привести к расходимости,
                слишком маленькое — к медленной сходимости.
            n_iterations: Количество итераций градиентного спуска.
                Должно быть положительным целым числом.

        Raises:
            TypeError: Если типы аргументов некорректны.
            ValueError: Если значения аргументов невалидны.
        """
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(
                f"learning_rate должен быть числом, "
                f"получен {type(learning_rate).__name__}"
            )
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate должен быть положительным, получено {learning_rate}"
            )

        if not isinstance(n_iterations, int):
            raise TypeError(
                f"n_iterations должен быть целым числом, "
                f"получен {type(n_iterations).__name__}"
            )
        if n_iterations <= 0:
            raise ValueError(
                f"n_iterations должен быть положительным, получено {n_iterations}"
            )

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights: NDArray[np.floating] | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> LinearRegressionGD:
        """
        Обучает модель на переданных данных методом градиентного спуска.
        Алгоритм:
            1. Инициализация весов нулями.
            2. На каждой итерации:
               a. Вычисление предсказаний: ŷ = X·w + b.
               b. Вычисление ошибки: e = ŷ − y.
               c. Вычисление градиентов:
                  ∂L/∂w = (2/n) · Xᵀ · e
                  ∂L/∂b = (2/n) · Σ e
               d. Обновление параметров.
               e. Запись значения функции потерь.

        Args:
            X: Матрица признаков размером (n_samples, n_features).
                Каждая строка — один объект, каждый столбец — один признак.
            y: Вектор целевых значений размером (n_samples,).

        Returns:
            Ссылка на обученную модель (self) для поддержки цепочек вызовов.

        Raises:
            TypeError: Если X или y не являются numpy-массивами.
            ValueError: Если размерности X и y несовместимы.
        """
        X, y = self._validate_input(X, y)

        n_samples, n_features = X.shape

        # Инициализация параметров нулями
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            # Прямой проход: вычисление предсказаний
            y_predicted = X @ self.weights + self.bias

            # Вычисление ошибки
            error = y_predicted - y

            # Вычисление градиентов
            dw = (2.0 / n_samples) * (X.T @ error)
            db = (2.0 / n_samples) * np.sum(error)

            # Обновление параметров (шаг градиентного спуска)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Сохранение значения функции потерь
            current_loss = float(np.mean(error**2))
            self.loss_history.append(current_loss)

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Вычисляет предсказания для переданных объектов.

        Args:
            X: Матрица признаков размером (n_samples, n_features).

        Returns:
            Вектор предсказаний размером (n_samples,).

        Raises:
            RuntimeError: Если модель ещё не обучена (не вызван fit).
            TypeError: Если X не является numpy-массивом.
            ValueError: Если количество признаков не совпадает с обученной моделью.
        """
        if self.weights is None:
            raise RuntimeError(
                "Модель ещё не обучена. Сначала вызовите метод fit()."
            )

        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"X должен быть numpy.ndarray, получен {type(X).__name__}"
            )

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != len(self.weights):
            raise ValueError(
                f"Ожидается {len(self.weights)} признаков, "
                f"получено {X.shape[1]}"
            )

        return X @ self.weights + self.bias

    @staticmethod
    def _validate_input(
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Проверяет корректность входных данных для обучения.

        Args:
            X: Матрица признаков.
            y: Вектор целевых значений.

        Returns:
            Кортеж (X, y) с корректными размерностями.

        Raises:
            TypeError: Если X или y не являются numpy-массивами.
            ValueError: Если размерности не согласованы.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"X должен быть numpy.ndarray, получен {type(X).__name__}"
            )
        if not isinstance(y, np.ndarray):
            raise TypeError(
                f"y должен быть numpy.ndarray, получен {type(y).__name__}"
            )

        # Если X — одномерный вектор, преобразуем в матрицу-столбец
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                f"X должен быть двумерным массивом, получена размерность {X.ndim}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"y должен быть одномерным вектором, получена размерность {y.ndim}"
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Количество объектов в X ({X.shape[0]}) "
                f"не совпадает с длиной y ({y.shape[0]})"
            )

        return X, y


def mse(y_true: NDArray[np.floating], y_pred: NDArray[np.floating]) -> float:
    """Вычисляет среднеквадратичную ошибку (Mean Squared Error).

    Формула:
        MSE = (1/n) · Σᵢ (yᵢ_true − yᵢ_pred)²

    Args:
        y_true: Вектор истинных значений размером (n_samples,).
        y_pred: Вектор предсказаний размером (n_samples,).

    Returns:
        Значение MSE (неотрицательное вещественное число).

    Raises:
        TypeError: Если аргументы не являются numpy-массивами.
        ValueError: Если длины векторов не совпадают.
    """
    if not isinstance(y_true, np.ndarray):
        raise TypeError(
            f"y_true должен быть numpy.ndarray, получен {type(y_true).__name__}"
        )
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f"y_pred должен быть numpy.ndarray, получен {type(y_pred).__name__}"
        )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Размерности y_true {y_true.shape} и y_pred {y_pred.shape} "
            f"не совпадают"
        )

    return float(np.mean((y_true - y_pred) ** 2))
