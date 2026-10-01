"""Графики в SVG: один стиль на все исследования, подписи по-русски.

SVG, а не PNG, по двум причинам: файл остаётся текстовым и читаемым в системе контроля версий, и он
не размывается при увеличении. Графики сохраняются рядом с отчётом, а не встраиваются в него
основанием 64 — из-за такого встраивания ноутбуки разрастаются до нескольких мегабайт.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")  # без окна: графики только в файлы

import matplotlib.pyplot as plt

INK: Final = "#1b1f24"
MUTED: Final = "#6b7683"
ACCENT: Final = "#2f6f4e"
SECOND: Final = "#a8451f"
GRID: Final = "#dfe3e8"
SIZE: Final = (8.0, 4.5)


def _figure(title: str, subtitle: str = "") -> tuple[Any, Any]:
    figure, axes = plt.subplots(figsize=SIZE)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    axes.set_title(title, color=INK, fontsize=13, pad=16 if subtitle else 10, loc="left")
    if subtitle:
        axes.text(0.0, 1.02, subtitle, transform=axes.transAxes, color=MUTED, fontsize=10)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=9)
    return figure, axes


def _save(figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return path


def bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    path: Path,
    title: str,
    subtitle: str = "",
    ylabel: str = "",
    horizontal: bool = False,
) -> Path:
    figure, axes = _figure(title, subtitle)
    if horizontal:
        axes.barh(list(labels), list(values), color=ACCENT)
        axes.invert_yaxis()
        axes.grid(axis="x", color=GRID, linewidth=0.8)
        axes.grid(axis="y", visible=False)
    else:
        axes.bar(list(labels), list(values), color=ACCENT)
        if max(len(label) for label in labels) > 8:
            plt.setp(axes.get_xticklabels(), rotation=30, ha="right")
    axes.set_ylabel(ylabel, color=MUTED, fontsize=10)
    return _save(figure, path)


def grouped_bars(
    labels: Sequence[str],
    first: Sequence[float],
    second: Sequence[float],
    *,
    names: tuple[str, str],
    path: Path,
    title: str,
    subtitle: str = "",
    ylabel: str = "",
) -> Path:
    figure, axes = _figure(title, subtitle)
    positions = range(len(labels))
    width = 0.38
    axes.bar([position - width / 2 for position in positions], list(first), width, label=names[0], color=ACCENT)
    axes.bar([position + width / 2 for position in positions], list(second), width, label=names[1], color=SECOND)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(list(labels))
    axes.set_ylabel(ylabel, color=MUTED, fontsize=10)
    axes.legend(frameon=False, fontsize=9)
    return _save(figure, path)


def lines(
    x: Sequence[Any],
    series: dict[str, Sequence[float]],
    *,
    path: Path,
    title: str,
    subtitle: str = "",
    ylabel: str = "",
) -> Path:
    figure, axes = _figure(title, subtitle)
    palette = [ACCENT, SECOND, MUTED, "#3a6ea5"]
    for number, (name, values) in enumerate(series.items()):
        axes.plot(list(x), list(values), label=name, color=palette[number % len(palette)], linewidth=2)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=10)
    if len(series) > 1:
        axes.legend(frameon=False, fontsize=9)
    if len(x) > 8:
        plt.setp(axes.get_xticklabels(), rotation=30, ha="right")
    return _save(figure, path)


def histogram(
    values: Sequence[float],
    *,
    path: Path,
    title: str,
    subtitle: str = "",
    bins: int = 30,
    xlabel: str = "",
) -> Path:
    figure, axes = _figure(title, subtitle)
    axes.hist(list(values), bins=bins, color=ACCENT, edgecolor="white", linewidth=0.5)
    axes.set_xlabel(xlabel, color=MUTED, fontsize=10)
    axes.set_ylabel("наблюдений", color=MUTED, fontsize=10)
    return _save(figure, path)


def boxes(
    groups: dict[str, Sequence[float]],
    *,
    path: Path,
    title: str,
    subtitle: str = "",
    ylabel: str = "",
) -> Path:
    figure, axes = _figure(title, subtitle)
    data = [list(values) for values in groups.values()]
    drawing = axes.boxplot(data, tick_labels=list(groups), patch_artist=True)
    for patch in drawing["boxes"]:
        patch.set_facecolor("#dceadf")
        patch.set_edgecolor(ACCENT)
    for whisker in drawing["whiskers"] + drawing["caps"]:
        whisker.set_color(ACCENT)
    for median in drawing["medians"]:
        median.set_color(SECOND)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=10)
    return _save(figure, path)


def scatter(
    x: Sequence[float],
    y: Sequence[float],
    *,
    path: Path,
    title: str,
    subtitle: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Path:
    figure, axes = _figure(title, subtitle)
    axes.scatter(list(x), list(y), s=12, color=ACCENT, alpha=0.45, edgecolors="none")
    axes.set_xlabel(xlabel, color=MUTED, fontsize=10)
    axes.set_ylabel(ylabel, color=MUTED, fontsize=10)
    return _save(figure, path)


def funnel(steps: Sequence[str], values: Sequence[int], *, path: Path, title: str, subtitle: str = "") -> Path:
    """Воронка: столбики по убыванию с подписью доли от первого шага."""
    figure, axes = _figure(title, subtitle)
    axes.barh(list(steps), list(values), color=ACCENT)
    axes.invert_yaxis()
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.grid(axis="y", visible=False)
    top = values[0] if values else 1
    for position, value in enumerate(values):
        axes.text(value, position, f"  {value} ({value / top:.0%})", va="center", color=MUTED, fontsize=9)
    axes.set_xlim(0, max(values) * 1.25 if values else 1)
    return _save(figure, path)
