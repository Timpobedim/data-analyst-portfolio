from __future__ import annotations

import numpy as np
import pytest

from analiz import stats


def sample(mean: float, spread: float, size: int, seed: int = 1) -> list[float]:
    generator = np.random.default_rng(seed)
    return [float(value) for value in generator.normal(mean, spread, size)]


def test_clear_difference_is_found() -> None:
    result = stats.compare_means(sample(110, 10, 200), sample(100, 10, 200, seed=2))
    assert result.significant
    assert result.effect is not None
    assert result.effect > 0.8
    assert "Различие есть" in result.conclusion


def test_identical_groups_show_no_difference() -> None:
    result = stats.compare_means(sample(100, 10, 200), sample(100, 10, 200, seed=3))
    assert not result.significant
    assert "Различий не обнаружено" in result.conclusion


def test_unequal_variances_switch_the_criterion() -> None:
    equal = stats.compare_means(sample(100, 10, 300), sample(101, 10, 300, seed=4))
    unequal = stats.compare_means(sample(100, 2, 300), sample(101, 30, 300, seed=5))
    assert equal.name == "критерий Стьюдента"
    assert unequal.name == "критерий Уэлча", "при разных дисперсиях нужен Уэлч, а не Стьюдент"


def test_tiny_groups_are_refused() -> None:
    with pytest.raises(ValueError, match="хотя бы два наблюдения"):
        stats.compare_means([1.0], [2.0, 3.0])


def test_distributions_are_compared_by_ranks() -> None:
    result = stats.compare_distributions([1, 2, 3, 4, 100, 200], [10, 11, 12, 13, 14, 15])
    assert result.name == "критерий Манна–Уитни"
    assert 0 <= result.p_value <= 1
    assert result.effect is not None


def test_shares_are_compared_with_interval() -> None:
    result = stats.compare_shares(150, 1000, 100, 1000, labels=("баннер", "контроль"))
    assert result.significant
    assert result.effect == pytest.approx(0.05)
    assert "доверительный интервал" in result.conclusion


def test_equal_shares_are_not_significant() -> None:
    assert not stats.compare_shares(100, 1000, 102, 1000).significant


@pytest.mark.parametrize(
    ("successes_a", "total_a", "successes_b", "total_b"),
    [(10, 0, 5, 100), (200, 100, 5, 100), (-1, 100, 5, 100)],
)
def test_impossible_counts_are_refused(successes_a: int, total_a: int, successes_b: int, total_b: int) -> None:
    with pytest.raises(ValueError, match=r"наблюдени|успехов"):
        stats.compare_shares(successes_a, total_a, successes_b, total_b)


def test_chi_square_finds_dependency() -> None:
    independent = stats.chi_square([[500, 500], [500, 500]])
    dependent = stats.chi_square([[900, 100], [100, 900]])
    assert not independent.significant
    assert dependent.significant
    assert dependent.effect is not None
    assert 0 <= dependent.effect <= 1


def test_chi_square_needs_a_table() -> None:
    with pytest.raises(ValueError, match="таблица сопряжённости"):
        stats.chi_square([[1, 2, 3]][0:0])


def test_holm_stops_at_the_first_failure() -> None:
    assert stats.holm([0.001, 0.04, 0.2]) == [True, False, False]
    assert stats.holm([0.001, 0.002, 0.003]) == [True, True, True]
    assert stats.holm([]) == []


def test_holm_keeps_the_order_of_input() -> None:
    decisions = stats.holm([0.2, 0.001])
    assert decisions == [False, True], "решения возвращаются в порядке исходных гипотез"


def test_cohen_d_has_sign_and_size() -> None:
    assert stats.cohen_d(sample(110, 10, 200), sample(100, 10, 200, seed=6)) > 0
    assert stats.cohen_d(sample(100, 10, 200), sample(110, 10, 200, seed=7)) < 0
    assert stats.cohen_d([1.0], [2.0]) == 0.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.05, "пренебрежимо малый"), (0.3, "небольшой"), (0.6, "средний"), (1.2, "большой")],
)
def test_effect_size_is_named(value: float, expected: str) -> None:
    assert stats.effect_size_name(value) == expected


def test_mean_interval_surrounds_the_mean() -> None:
    values = sample(100, 10, 500)
    low, high = stats.mean_interval(values)
    assert low < float(np.mean(values)) < high
    wide = stats.mean_interval(values[:50])
    assert (wide[1] - wide[0]) > (high - low), "на меньшей выборке интервал шире"


def test_share_interval_stays_inside_bounds() -> None:
    low, high = stats.share_interval(0, 50)
    # Формула Уилсона даёт не ровно ноль, а машинный ноль — это нормально для плавающей точки.
    assert low == pytest.approx(0.0, abs=1e-9)
    assert 0 < high < 0.2
    assert stats.share_interval(50, 50)[1] == 1.0


def test_required_sample_grows_when_effect_shrinks() -> None:
    small_lift = stats.required_sample(0.1, 0.005)
    big_lift = stats.required_sample(0.1, 0.05)
    assert small_lift > big_lift
    assert big_lift > 0


@pytest.mark.parametrize(("baseline", "lift"), [(0.0, 0.01), (1.0, 0.01), (0.1, 0.0), (0.1, -0.2)])
def test_required_sample_refuses_nonsense(baseline: float, lift: float) -> None:
    with pytest.raises(ValueError, match=r"доля|прирост"):
        stats.required_sample(baseline, lift)


def test_result_line_mentions_the_verdict() -> None:
    result = stats.compare_shares(150, 1000, 100, 1000)
    assert "p =" in result.line()
    assert "есть различие" in result.line()
