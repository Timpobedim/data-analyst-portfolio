"""Командная строка: list, run, all.

Все восемь исследований запускаются одной командой и пересобирают свои отчёты целиком. Это и есть
ответ на главную претензию к портфолио из ноутбуков: здесь нельзя «забыть перезапустить ячейку» —
отчёт либо собирается из данных сейчас, либо не собирается вовсе.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from analiz import data
from analiz.projects import Project, find, registry
from analiz.report import number

DEFAULT_OUT = "проекты"


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_console()
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


def _utf8_console() -> None:
    """В стандартной консоли Windows нет знака рубля и кавычек-ёлочек: переключаем вывод на UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() not in {"utf-8", "utf8"}:
            stream.reconfigure(encoding="utf-8", errors="replace")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analiz", description="Восемь аналитических исследований")
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="список исследований")
    listing.set_defaults(handler=_list)

    single = commands.add_parser("run", help="пересобрать одно исследование")
    single.add_argument("name", help="имя исследования из «analiz list»")
    _common(single)
    single.set_defaults(handler=_run)

    every = commands.add_parser("all", help="пересобрать все исследования")
    _common(every)
    every.set_defaults(handler=_all)
    return parser


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT), help="куда складывать отчёты")
    parser.add_argument("--seed", type=int, default=data.SEED, help="зерно генератора данных")


def _list(_: argparse.Namespace) -> int:
    for project in registry():
        print(f"{project.name:15s} {project.title}")
        print(f"{'':15s} {project.question}")
    return 0


def _run(args: argparse.Namespace) -> int:
    project = find(args.name)
    if project is None:
        known = ", ".join(item.name for item in registry())
        raise ValueError(f"Исследования «{args.name}» нет. Есть: {known}")
    _build(project, Path(args.out), args.seed)
    return 0


def _all(args: argparse.Namespace) -> int:
    started = time.monotonic()
    for project in registry():
        _build(project, Path(args.out), args.seed)
    print(f"\nГотово: {len(registry())} исследований за {time.monotonic() - started:.1f} с.")
    return 0


def _build(project: Project, out: Path, seed: int) -> None:
    folder = out / project.folder
    started = time.monotonic()
    report = project.run(folder, seed)
    path = report.save(folder / "отчёт.md")
    charts_count = len(list((folder / "графики").glob("*.svg")))
    print(
        f"+ {project.name:15s} {path} — разделов {len(report.sections)}, "
        f"графиков {charts_count}, выводов {len(report.findings)}, "
        f"за {number(time.monotonic() - started, 1)} с"
    )


if __name__ == "__main__":
    raise SystemExit(main())
