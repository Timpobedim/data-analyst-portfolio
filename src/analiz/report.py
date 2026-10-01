"""Отчёт: вопрос — данные — метод — что получилось — чего это не доказывает.

Структура одинаковая у всех восьми исследований, и это не формальность. Читатель отчёта обычно ищет
ответ на один вопрос и хочет понимать, насколько ему можно верить. Поэтому в каждом отчёте есть
раздел «чего это не доказывает»: без него любая аналитика читается увереннее, чем заслуживает.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Section:
    """Раздел отчёта: текст, при необходимости таблица и график."""

    title: str
    text: str = ""
    headers: Sequence[str] = ()
    rows: Sequence[Sequence[str]] = ()
    image: Path | None = None
    caption: str = ""


@dataclass
class Report:
    title: str
    question: str
    data: str
    method: str
    sections: list[Section] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)
    prepared_on: date | None = None

    def add(self, section: Section) -> None:
        self.sections.append(section)

    def to_markdown(self, *, image_prefix: str = "графики") -> str:
        parts = [f"# {self.title}", ""]
        parts += [f"**Вопрос:** {self.question}", "", f"**Данные:** {self.data}", "", f"**Метод:** {self.method}", ""]
        for section in self.sections:
            parts.append(f"## {section.title}")
            parts.append("")
            if section.text:
                parts += [section.text, ""]
            if section.headers and section.rows:
                parts += [table(section.headers, section.rows), ""]
            if section.image is not None:
                caption = section.caption or section.title
                parts += [f"![{caption}]({image_prefix}/{section.image.name})", ""]
        if self.findings:
            parts += ["## Что получилось", ""]
            parts += [f"{number}. {text}" for number, text in enumerate(self.findings, start=1)]
            parts.append("")
        if self.limits:
            parts += ["## Чего это не доказывает", ""]
            parts += [f"- {text}" for text in self.limits]
            parts.append("")
        if self.prepared_on:
            parts += [
                f"*Отчёт собран {self.prepared_on:%d.%m.%Y} командой `analiz`; данные воспроизводимы по зерну.*",
                "",
            ]
        return "\n".join(parts)

    def save(self, path: Path, *, image_prefix: str = "графики") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(image_prefix=image_prefix), "utf-8")
        return path


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Таблица в Markdown: её видно и в репозитории, и в любом просмотрщике."""
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def number(value: float, digits: int = 0) -> str:
    """1234567.8 → «1 234 568». Разделитель разрядов — обычный пробел, чтобы не ломать вывод в консоли."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def money(value: float, digits: int = 0) -> str:
    return f"{number(value, digits)} ₽"


def percent(value: float, digits: int = 1) -> str:
    """0.1234 → «12,3%»."""
    return f"{value * 100:.{digits}f}".replace(".", ",") + "%"


def change(value: float, digits: int = 1) -> str:
    """Изменение со знаком: «+12,3%»."""
    sign = "+" if value >= 0 else ""
    return sign + percent(value, digits)


def as_float(value: object) -> float:
    """pandas отдаёт значения ячеек с очень широким типом; приводим их к числу в одном месте."""
    if isinstance(value, int | float):
        return float(value)
    return float(str(value))
