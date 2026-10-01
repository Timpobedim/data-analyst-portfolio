from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from analiz.report import Report, Section, change, money, number, percent, table


def sample_report() -> Report:
    report = Report(
        title="Пробное исследование",
        question="Работает ли отчёт?",
        data="10 строк",
        method="проверка",
        prepared_on=date(2026, 3, 10),
    )
    report.add(
        Section(
            title="Что пришло",
            text="Данные в порядке.",
            headers=["Поле", "Пропусков"],
            rows=[["цена", "0"]],
            image=Path("графики/пример.svg"),
        )
    )
    report.findings = ["Первый вывод", "Второй вывод"]
    report.limits = ["Это синтетические данные"]
    return report


def test_markdown_has_every_part() -> None:
    text = sample_report().to_markdown()
    assert text.startswith("# Пробное исследование")
    assert "**Вопрос:** Работает ли отчёт?" in text
    assert "## Что пришло" in text
    assert "| Поле | Пропусков |" in text
    assert "![Что пришло](графики/пример.svg)" in text
    assert "1. Первый вывод" in text
    assert "## Чего это не доказывает" in text
    assert "- Это синтетические данные" in text
    assert "10.03.2026" in text


def test_report_without_findings_skips_the_section() -> None:
    report = Report(title="Пусто", question="?", data="нет", method="нет")
    text = report.to_markdown()
    assert "## Что получилось" not in text
    assert "## Чего это не доказывает" not in text


def test_report_is_saved(tmp_path: Path) -> None:
    path = sample_report().save(tmp_path / "отчёт.md")
    assert path.exists()
    assert "Пробное исследование" in path.read_text("utf-8")


def test_table_renders_markdown() -> None:
    rendered = table(["Город", "Цена"], [["Москва", "100"], ["Тверь", "90"]])
    lines = rendered.splitlines()
    assert lines[0] == "| Город | Цена |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| Москва | 100 |"


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [(1234567.0, 0, "1 234 567"), (1234.56, 2, "1 234,56"), (0.0, 0, "0")],
)
def test_numbers_are_readable(value: float, digits: int, expected: str) -> None:
    assert number(value, digits) == expected


def test_money_and_percent() -> None:
    assert money(1234567.0) == "1 234 567 ₽"
    assert percent(0.1234) == "12,3%"
    assert percent(0.5, 0) == "50%"
    assert change(0.123) == "+12,3%"
    assert change(-0.05) == "-5,0%"
