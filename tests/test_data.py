from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from analiz import data


def test_same_seed_gives_the_same_numbers() -> None:
    first = data.generator(7).normal(0, 1, 100)
    second = data.generator(7).normal(0, 1, 100)
    assert list(first) == list(second), "воспроизводимость — главное требование к данным исследования"
    assert list(data.generator(8).normal(0, 1, 100)) != list(first)


def test_days_and_months_walk_forward() -> None:
    assert data.days(date(2026, 1, 30), 3) == [date(2026, 1, 30), date(2026, 1, 31), date(2026, 2, 1)]
    assert data.months(date(2025, 11, 1), 4) == [
        date(2025, 11, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 2, 1),
    ]


def test_money_is_rounded_to_kopecks() -> None:
    assert list(data.money([1.239, 2.005])) == [1.24, 2.0]


def test_missing_values_are_added_in_the_right_share() -> None:
    frame = pd.DataFrame({"цена": list(range(100))})
    spoiled = data.spoil_missing(frame, "цена", 0.1, data.generator(1))
    assert int(spoiled["цена"].isna().sum()) == 10
    assert int(frame["цена"].isna().sum()) == 0, "исходный набор не меняется"


def test_duplicates_are_added() -> None:
    frame = pd.DataFrame({"a": list(range(100))})
    spoiled = data.spoil_duplicates(frame, 0.05, data.generator(1))
    assert len(spoiled) == 105
    assert int(spoiled.duplicated().sum()) == 5


def test_outliers_are_multiplied() -> None:
    frame = pd.DataFrame({"цена": [100.0] * 100})
    spoiled = data.spoil_outliers(frame, "цена", 0.03, 50.0, data.generator(2))
    assert int((spoiled["цена"] > 1_000).sum()) == 3


def test_csv_round_trip_keeps_russian_headers(tmp_path: Path) -> None:
    frame = pd.DataFrame({"город": ["Москва", "Тверь"], "цена": [1_200.5, 990.0]})
    path = data.save_csv(frame, tmp_path / "данные.csv")
    assert path.exists()

    text = path.read_text("utf-8-sig")
    assert "город;цена" in text, "разделитель — точка с запятой: так файл открывается в русском Excel"

    restored = data.read_csv(path)
    assert list(restored.columns) == ["город", "цена"]
    assert restored["город"].iloc[0] == "Москва"
