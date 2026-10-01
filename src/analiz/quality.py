"""Проверка данных до анализа: пропуски, дубли, выбросы и диапазоны.

Это первый раздел каждого отчёта, и он не для красоты. Вывод «средний чек вырос на 12%» ничего не
стоит, если в данных лежали дубли выгрузки, а половина строк пришла с пустым городом. Поэтому
сначала — что с данными не так, потом — что с ними сделали, и только потом выводы.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

IQR_FACTOR: Final = 1.5


@dataclass(frozen=True, slots=True)
class ColumnQuality:
    name: str
    kind: str
    missing: int
    missing_share: float
    unique: int
    example: str


@dataclass(frozen=True, slots=True)
class Quality:
    """Что известно о наборе до того, как его начали считать."""

    rows: int
    duplicates: int
    columns: tuple[ColumnQuality, ...]

    @property
    def has_problems(self) -> bool:
        return self.duplicates > 0 or any(column.missing for column in self.columns)

    def table(self) -> tuple[list[str], list[list[str]]]:
        headers = ["Поле", "Тип", "Пропусков", "Доля пропусков", "Уникальных", "Пример"]
        rows = [
            [
                column.name,
                column.kind,
                str(column.missing),
                f"{column.missing_share:.1%}",
                str(column.unique),
                column.example,
            ]
            for column in self.columns
        ]
        return headers, rows

    def summary(self) -> str:
        if not self.has_problems:
            return f"Строк {self.rows}, пропусков и дублей нет."
        parts = [f"строк {self.rows}"]
        if self.duplicates:
            parts.append(f"полных дублей {self.duplicates}")
        missing = [column for column in self.columns if column.missing]
        if missing:
            listed = ", ".join(f"{column.name} — {column.missing_share:.1%}" for column in missing)
            parts.append(f"пропуски: {listed}")
        return "Исходные данные: " + "; ".join(parts) + "."


@dataclass(frozen=True, slots=True)
class Outliers:
    low: float
    high: float
    count: int
    share: float

    def line(self, column: str) -> str:
        return (
            f"«{column}»: обычные значения от {self.low:,.0f} до {self.high:,.0f}".replace(",", " ")
            + f", за границами {self.count} наблюдений ({self.share:.1%})"
        )


@dataclass(frozen=True, slots=True)
class CleanReport:
    """Что именно сделали с данными перед анализом — чтобы это можно было оспорить."""

    dropped_duplicates: int
    dropped_missing: int
    dropped_outliers: int
    rows_before: int
    rows_after: int

    def line(self) -> str:
        parts = []
        if self.dropped_duplicates:
            parts.append(f"убрано дублей {self.dropped_duplicates}")
        if self.dropped_missing:
            parts.append(f"строк с пропусками {self.dropped_missing}")
        if self.dropped_outliers:
            parts.append(f"выбросов {self.dropped_outliers}")
        if not parts:
            return f"Данные не менялись: осталось {self.rows_after} строк."
        return f"Подготовка: {', '.join(parts)}; было {self.rows_before} строк, стало {self.rows_after}."


def inspect(frame: pd.DataFrame) -> Quality:
    """Собирает картину исходных данных: типы, пропуски, уникальные значения."""
    rows = len(frame)
    columns = []
    for name in frame.columns:
        series = frame[name]
        missing = int(series.isna().sum())
        present = series.dropna()
        example = str(present.iloc[0]) if len(present) else "—"
        columns.append(
            ColumnQuality(
                name=str(name),
                kind=str(series.dtype),
                missing=missing,
                missing_share=missing / rows if rows else 0.0,
                unique=int(series.nunique(dropna=True)),
                example=example[:28],
            )
        )
    return Quality(rows=rows, duplicates=int(frame.duplicated().sum()), columns=tuple(columns))


def outliers(values: Sequence[float] | pd.Series, factor: float = IQR_FACTOR) -> Outliers:
    """Границы по межквартильному размаху: полтора размаха в обе стороны."""
    sample = np.asarray(pd.Series(values).dropna(), dtype=float)
    if sample.size == 0:
        return Outliers(low=0.0, high=0.0, count=0, share=0.0)
    first, third = np.percentile(sample, [25, 75])
    spread = third - first
    low = float(first - factor * spread)
    high = float(third + factor * spread)
    beyond = int(((sample < low) | (sample > high)).sum())
    return Outliers(low=low, high=high, count=beyond, share=beyond / sample.size)


def clean(
    frame: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    required: Sequence[str] = (),
    trim: Sequence[str] = (),
    factor: float = IQR_FACTOR,
) -> tuple[pd.DataFrame, CleanReport]:
    """Готовит данные и честно рассказывает, сколько строк потеряно и почему.

    `required` — поля, без которых строка бесполезна; `trim` — числовые поля, где убираются выбросы.
    """
    before = len(frame)
    result = frame.copy()

    dropped_duplicates = 0
    if drop_duplicates:
        dropped_duplicates = int(result.duplicated().sum())
        result = result.drop_duplicates().reset_index(drop=True)

    dropped_missing = 0
    if required:
        was = len(result)
        result = result.dropna(subset=list(required)).reset_index(drop=True)
        dropped_missing = was - len(result)

    dropped_outliers = 0
    for column in trim:
        bounds = outliers(result[column], factor)
        was = len(result)
        result = result[(result[column] >= bounds.low) & (result[column] <= bounds.high)].reset_index(drop=True)
        dropped_outliers += was - len(result)

    return result, CleanReport(
        dropped_duplicates=dropped_duplicates,
        dropped_missing=dropped_missing,
        dropped_outliers=dropped_outliers,
        rows_before=before,
        rows_after=len(result),
    )
