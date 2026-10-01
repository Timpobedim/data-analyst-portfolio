"""Проверка гипотез так, чтобы вывод можно было защитить.

Три вещи, без которых цифры не значат ничего, и которых чаще всего не хватает в учебных работах:

1. **Размер эффекта.** При выборке в сто тысяч строк значимым становится любое различие. «p < 0,05»
   без величины эффекта не отвечает на вопрос «а это вообще много?».
2. **Поправка на множественные сравнения.** Пятнадцать проверок с уровнем 0,05 дают в среднем
   одну «находку» на пустом месте. Здесь для набора гипотез применяется поправка Холма.
3. **Честный выбор критерия.** Сначала проверяется равенство дисперсий, и в зависимости от этого
   берётся Стьюдент или Уэлч; для сильно скошенных данных — Манна–Уитни.

Каждая проверка возвращает не только p-значение, но и готовую формулировку вывода на русском.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy import stats as scipy_stats

ALPHA: Final = 0.05
SMALL: Final = 0.2
MEDIUM: Final = 0.5
LARGE: Final = 0.8


@dataclass(frozen=True, slots=True)
class TestResult:
    """Результат одной проверки: что считали, что получилось и что это значит."""

    name: str
    statistic: float
    p_value: float
    alpha: float
    effect: float | None
    effect_name: str
    conclusion: str

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def line(self) -> str:
        mark = "есть различие" if self.significant else "различий не видно"
        effect = f", {self.effect_name} {self.effect:.2f}" if self.effect is not None else ""
        return f"{self.name}: p = {self.p_value:.4f} ({mark}){effect}"


def compare_means(
    first: Sequence[float],
    second: Sequence[float],
    *,
    alpha: float = ALPHA,
    labels: tuple[str, str] = ("первая группа", "вторая группа"),
) -> TestResult:
    """Сравнение средних: критерий Левена решает, брать Стьюдента или Уэлча."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    _check_sizes(left, right)

    equal_variance = float(scipy_stats.levene(left, right).pvalue) > alpha
    result = scipy_stats.ttest_ind(left, right, equal_var=equal_variance)
    effect = cohen_d(left, right)
    name = "критерий Стьюдента" if equal_variance else "критерий Уэлча"
    difference = float(np.mean(left) - np.mean(right))
    conclusion = _conclusion(
        significant=float(result.pvalue) < alpha,
        what=f"средние {labels[0]} и {labels[1]}",
        difference=f"разница средних {difference:+.2f}",
        effect=effect,
    )
    return TestResult(
        name=name,
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alpha=alpha,
        effect=effect,
        effect_name="размер эффекта d",
        conclusion=conclusion,
    )


def compare_distributions(
    first: Sequence[float],
    second: Sequence[float],
    *,
    alpha: float = ALPHA,
    labels: tuple[str, str] = ("первая группа", "вторая группа"),
) -> TestResult:
    """Манна–Уитни: когда данные скошены и среднее вводит в заблуждение."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    _check_sizes(left, right)

    result = scipy_stats.mannwhitneyu(left, right, alternative="two-sided")
    effect = rank_biserial(left, right)
    conclusion = _conclusion(
        significant=float(result.pvalue) < alpha,
        what=f"распределения {labels[0]} и {labels[1]}",
        difference=f"медианы {float(np.median(left)):.2f} и {float(np.median(right)):.2f}",
        effect=effect,
    )
    return TestResult(
        name="критерий Манна–Уитни",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alpha=alpha,
        effect=effect,
        effect_name="ранговая корреляция",
        conclusion=conclusion,
    )


def compare_shares(
    successes_first: int,
    total_first: int,
    successes_second: int,
    total_second: int,
    *,
    alpha: float = ALPHA,
    labels: tuple[str, str] = ("первая группа", "вторая группа"),
) -> TestResult:
    """Сравнение долей: z-критерий для двух пропорций и доверительный интервал их разницы."""
    if total_first <= 0 or total_second <= 0:
        raise ValueError("В каждой группе должно быть хотя бы одно наблюдение")
    if not 0 <= successes_first <= total_first or not 0 <= successes_second <= total_second:
        raise ValueError("Число успехов не может быть больше числа наблюдений")

    share_first = successes_first / total_first
    share_second = successes_second / total_second
    pooled = (successes_first + successes_second) / (total_first + total_second)
    standard = math.sqrt(pooled * (1 - pooled) * (1 / total_first + 1 / total_second))
    if standard == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (share_first - share_second) / standard
        p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(statistic))))

    low, high = share_difference_interval(successes_first, total_first, successes_second, total_second)
    conclusion = _conclusion(
        significant=p_value < alpha,
        what=f"доли {labels[0]} и {labels[1]}",
        difference=f"{share_first:.2%} против {share_second:.2%}, разница {share_first - share_second:+.2%} "
        f"(доверительный интервал от {low:+.2%} до {high:+.2%})",
        effect=None,
    )
    return TestResult(
        name="z-критерий для долей",
        statistic=float(statistic),
        p_value=p_value,
        alpha=alpha,
        effect=share_first - share_second,
        effect_name="разница долей",
        conclusion=conclusion,
    )


def chi_square(table: Sequence[Sequence[int]], *, alpha: float = ALPHA, what: str = "признаки") -> TestResult:
    """Хи-квадрат для таблицы сопряжённости плюс V Крамера как размер эффекта."""
    matrix = np.asarray(table, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("Нужна таблица сопряжённости хотя бы 2×2")
    result = scipy_stats.chi2_contingency(matrix)
    total = float(matrix.sum())
    smaller = min(matrix.shape) - 1
    effect = math.sqrt(float(result.statistic) / (total * smaller)) if total and smaller else 0.0
    conclusion = _conclusion(
        significant=float(result.pvalue) < alpha,
        what=f"связь между {what}",
        difference="таблица сопряжённости",
        effect=effect,
    )
    return TestResult(
        name="критерий хи-квадрат",
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        alpha=alpha,
        effect=effect,
        effect_name="V Крамера",
        conclusion=conclusion,
    )


def holm(p_values: Sequence[float], *, alpha: float = ALPHA) -> list[bool]:
    """Поправка Холма: какие из нескольких проверок остаются значимыми.

    Без неё пятнадцать независимых проверок с уровнем 0,05 почти наверняка дадут ложную находку.
    """
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    decisions = [False] * len(p_values)
    total = len(p_values)
    for position, index in enumerate(order):
        threshold = alpha / (total - position)
        if p_values[index] < threshold:
            decisions[index] = True
        else:
            break  # как только одна проверка не прошла, остальные тоже не проходят
    return decisions


def cohen_d(first: Sequence[float] | np.ndarray[Any, Any], second: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Размер эффекта: на сколько стандартных отклонений различаются средние."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.size < 2 or right.size < 2:
        return 0.0
    pooled = math.sqrt(
        ((left.size - 1) * left.var(ddof=1) + (right.size - 1) * right.var(ddof=1)) / (left.size + right.size - 2)
    )
    if pooled == 0:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def rank_biserial(
    first: Sequence[float] | np.ndarray[Any, Any], second: Sequence[float] | np.ndarray[Any, Any]
) -> float:
    """Размер эффекта для Манна–Уитни: от −1 до 1."""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.size == 0 or right.size == 0:
        return 0.0
    statistic = float(scipy_stats.mannwhitneyu(left, right, alternative="two-sided").statistic)
    return float(2 * statistic / (left.size * right.size) - 1)


def effect_size_name(value: float) -> str:
    """Человеческая оценка величины эффекта."""
    size = abs(value)
    if size < SMALL:
        return "пренебрежимо малый"
    if size < MEDIUM:
        return "небольшой"
    if size < LARGE:
        return "средний"
    return "большой"


def mean_interval(values: Sequence[float], *, level: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал среднего."""
    sample = np.asarray(values, dtype=float)
    if sample.size < 2:
        raise ValueError("Для интервала нужно хотя бы два наблюдения")
    error = float(scipy_stats.sem(sample))
    margin = error * float(scipy_stats.t.ppf((1 + level) / 2, sample.size - 1))
    center = float(sample.mean())
    return center - margin, center + margin


def share_interval(successes: int, total: int, *, level: float = 0.95) -> tuple[float, float]:
    """Доверительный интервал доли по Уилсону — он не выходит за границы 0 и 1 на малых выборках."""
    if total <= 0:
        raise ValueError("Наблюдений должно быть больше нуля")
    share = successes / total
    z = float(scipy_stats.norm.ppf((1 + level) / 2))
    denominator = 1 + z**2 / total
    center = (share + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(share * (1 - share) / total + z**2 / (4 * total**2)) / denominator
    return max(center - margin, 0.0), min(center + margin, 1.0)


def share_difference_interval(
    successes_first: int, total_first: int, successes_second: int, total_second: int, *, level: float = 0.95
) -> tuple[float, float]:
    first = successes_first / total_first
    second = successes_second / total_second
    z = float(scipy_stats.norm.ppf((1 + level) / 2))
    error = math.sqrt(first * (1 - first) / total_first + second * (1 - second) / total_second)
    difference = first - second
    return difference - z * error, difference + z * error


def required_sample(baseline: float, lift: float, *, alpha: float = ALPHA, power: float = 0.8) -> int:
    """Сколько наблюдений в группе нужно, чтобы заметить прирост доли.

    Считается до эксперимента, а не после: A/B-тест, начатый без этой оценки, чаще всего заканчивается
    выводом «различий не обнаружено», который ничего не означает.
    """
    if not 0 < baseline < 1:
        raise ValueError("Базовая доля должна быть между нулём и единицей")
    if lift <= 0:
        raise ValueError("Ожидаемый прирост должен быть положительным")
    target = min(baseline + lift, 0.999999)
    z_alpha = float(scipy_stats.norm.ppf(1 - alpha / 2))
    z_power = float(scipy_stats.norm.ppf(power))
    average = (baseline + target) / 2
    numerator = (
        z_alpha * math.sqrt(2 * average * (1 - average))
        + z_power * math.sqrt(baseline * (1 - baseline) + target * (1 - target))
    ) ** 2
    return math.ceil(numerator / lift**2)


def _check_sizes(first: np.ndarray, second: np.ndarray) -> None:
    if first.size < 2 or second.size < 2:
        raise ValueError("В каждой группе должно быть хотя бы два наблюдения")


def _conclusion(*, significant: bool, what: str, difference: str, effect: float | None) -> str:
    if not significant:
        return f"Различий не обнаружено: {what} статистически неразличимы ({difference})."
    tail = f"; эффект {effect_size_name(effect)} ({effect:+.2f})" if effect is not None else ""
    return f"Различие есть: {what} отличаются ({difference}){tail}."
