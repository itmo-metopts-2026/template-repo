import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Фиксирует глобальное состояние всех генераторов случайных чисел.
    Устанавливает seed для следующих источников случайности:
        - встроенный модуль ``random`` (стандартная библиотека Python);
        - библиотека ``numpy`` (генератор ``numpy.random``);
        - переменная окружения ``PYTHONHASHSEED`` (детерминированное хеширование).

    Args:
        seed: Целое неотрицательное число, используемое в качестве начального
            значения генераторов. Рекомендуемое значение по умолчанию — 42.

    Raises:
        TypeError: Если ``seed`` не является целым числом.
        ValueError: Если ``seed`` отрицательный.

    Examples:
        >>> set_global_seed(42)
        >>> random.random()  # Результат будет воспроизводим
        0.6394267984578837
    """
    if not isinstance(seed, int):
        raise TypeError(
            f"Параметр seed должен быть целым числом, получен {type(seed).__name__}"
        )
    if seed < 0:
        raise ValueError(
            f"Параметр seed должен быть неотрицательным, получено {seed}"
        )

    random.seed(seed)

    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
