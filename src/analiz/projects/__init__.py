"""Реестр исследований: восемь работ, у каждой одинаковый вход и одинаковый выход.

Одинаковый порядок — не бюрократия: он позволяет прогнать все восемь одной командой, сравнивать их
между собой и держать отчёты в одном виде. Каждое исследование получает папку и возвращает отчёт.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from analiz.report import Report

Runner = Callable[[Path, int], Report]


@dataclass(frozen=True, slots=True)
class Project:
    """Одно исследование: имя для команды, папка для результатов и функция, которая его делает."""

    name: str
    title: str
    folder: str
    question: str
    run: Runner


def registry() -> tuple[Project, ...]:
    """Все исследования по порядку. Импорт внутри функции — чтобы реестр не тянул тяжёлые модули зря."""
    from analiz.projects import (
        boardgames,
        bookstore,
        carsharing,
        churn,
        delivery_ab,
        rent,
        school_funnel,
        scoring,
    )

    return (
        scoring.PROJECT,
        rent.PROJECT,
        carsharing.PROJECT,
        boardgames.PROJECT,
        delivery_ab.PROJECT,
        school_funnel.PROJECT,
        churn.PROJECT,
        bookstore.PROJECT,
    )


def find(name: str) -> Project | None:
    return next((project for project in registry() if project.name == name), None)
