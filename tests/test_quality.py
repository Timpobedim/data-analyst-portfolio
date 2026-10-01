from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analiz import quality


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "город": ["Москва", "Тверь", "Москва", "Тверь", None],
            "цена": [100.0, 120.0, 100.0, 5_000.0, 130.0],
            "комнат": [1, 2, 1, 2, 3],
        }
    )


def test_inspect_counts_rows_duplicates_and_gaps() -> None:
    checked = quality.inspect(frame())
    assert checked.rows == 5
    assert checked.duplicates == 1, "третья строка повторяет первую"
    city = next(column for column in checked.columns if column.name == "город")
    assert city.missing == 1
    assert city.missing_share == pytest.approx(0.2)
    assert city.unique == 2
    assert checked.has_problems


def test_clean_table_is_ready_for_the_report() -> None:
    headers, rows = quality.inspect(frame()).table()
    assert headers[0] == "Поле"
    assert len(rows) == 3
    assert any("20,0%" in cell or "20.0%" in cell for row in rows for cell in row)


def test_perfect_data_has_no_complaints() -> None:
    checked = quality.inspect(pd.DataFrame({"a": [1, 2, 3]}))
    assert not checked.has_problems
    assert "пропусков и дублей нет" in checked.summary()


def test_outliers_are_found_by_the_spread() -> None:
    values = [10.0] * 50 + [11.0] * 50 + [900.0]
    found = quality.outliers(values)
    assert found.count == 1
    assert found.high < 900
    assert found.share == pytest.approx(1 / 101, abs=0.001)
    assert "выбросы" not in found.line("цена")
    assert "цена" in found.line("цена")


def test_outliers_on_empty_data_are_zero() -> None:
    found = quality.outliers([])
    assert found.count == 0
    assert found.share == 0.0


def test_clean_removes_duplicates_and_gaps_and_outliers() -> None:
    """На трёх строках межквартильный размах выброса не видит — поэтому набор здесь побольше."""
    wide = pd.DataFrame({"город": ["Москва"] * 60, "цена": [100.0 + index * 0.1 for index in range(60)]})
    extra = pd.DataFrame({"город": [None, "Тверь"], "цена": [104.0, 9_000.0]})
    spoiled = pd.concat([wide, wide.iloc[[0]], extra], ignore_index=True)

    cleaned, report = quality.clean(spoiled, required=["город"], trim=["цена"])

    assert report.rows_before == 63
    assert report.dropped_duplicates == 1
    assert report.dropped_missing == 1
    assert report.dropped_outliers == 1, "9 000 при ценах около 100 — выброс"
    assert len(cleaned) == report.rows_after == 60
    assert cleaned["город"].isna().sum() == 0
    assert "Подготовка" in report.line()


def test_clean_without_work_says_so() -> None:
    clean_frame = pd.DataFrame({"a": [1, 2, 3]})
    result, report = quality.clean(clean_frame)
    assert len(result) == 3
    assert "не менялись" in report.line()


def test_clean_does_not_touch_the_original() -> None:
    original = frame()
    quality.clean(original, required=["город"])
    assert len(original) == 5, "исходный набор должен остаться нетронутым"


def test_outliers_ignore_missing_values() -> None:
    found = quality.outliers(pd.Series([1.0, 2.0, np.nan, 3.0, 100.0]))
    assert found.count == 1
