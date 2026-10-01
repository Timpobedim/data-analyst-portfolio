"""Воспроизводимые данные: одно зерно — один и тот же набор.

Данные для исследований генерируются здесь, а не скачиваются со сторонней платформы: так любой
может запустить исследование у себя. Команда `analiz`
создаёт данные сама, кладёт их рядом с отчётом в CSV и получает при каждом запуске ровно те же числа.

Данные намеренно «грязные»: пропуски, дубли и выбросы добавляются специально — именно с ними
и приходится работать в жизни, и именно на них проверяется, что подготовка данных написана честно.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

SEED: Final = 42
CSV_SEPARATOR: Final = ";"
CSV_ENCODING: Final = "utf-8-sig"
"""Метка порядка байтов и точка с запятой: так файл открывается в русском Excel без плясок с кодировкой."""


def generator(seed: int = SEED) -> np.random.Generator:
    """Генератор случайных чисел с зафиксированным зерном."""
    return np.random.default_rng(seed)


def days(start: date, count: int) -> list[date]:
    return [start + timedelta(days=number) for number in range(count)]


def months(start: date, count: int) -> list[date]:
    """Первые числа месяцев подряд."""
    out: list[date] = []
    year, month = start.year, start.month
    for _ in range(count):
        out.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return out


def money(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Суммы округляются до копеек: деньги не бывают с шестью знаками после запятой."""
    return np.round(np.asarray(values, dtype=float), 2)


def spoil_missing(frame: pd.DataFrame, column: str, share: float, rng: np.random.Generator) -> pd.DataFrame:
    """Проставляет пропуски в столбце — как в выгрузке, где поле необязательное."""
    spoiled = frame.copy()
    count = int(len(spoiled) * share)
    if count:
        rows = rng.choice(len(spoiled), size=count, replace=False)
        spoiled.loc[spoiled.index[rows], column] = None
    return spoiled


def spoil_duplicates(frame: pd.DataFrame, share: float, rng: np.random.Generator) -> pd.DataFrame:
    """Добавляет полные дубли строк — обычное дело при повторной выгрузке."""
    count = int(len(frame) * share)
    if not count:
        return frame.copy()
    rows = rng.choice(len(frame), size=count, replace=False)
    return pd.concat([frame, frame.iloc[rows]], ignore_index=True)


def spoil_outliers(
    frame: pd.DataFrame, column: str, share: float, factor: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Портит часть значений в несколько раз — опечатки оператора и единицы измерения."""
    spoiled = frame.copy()
    count = int(len(spoiled) * share)
    if count:
        rows = rng.choice(len(spoiled), size=count, replace=False)
        index = spoiled.index[rows]
        spoiled.loc[index, column] = spoiled.loc[index, column] * factor
    return spoiled


def save_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Сохраняет набор рядом с отчётом: любой читатель может открыть его и проверить выводы сам."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=CSV_SEPARATOR, encoding=CSV_ENCODING, index=False)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
