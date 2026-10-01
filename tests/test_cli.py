from __future__ import annotations

from pathlib import Path

import pytest

from analiz.cli import main
from analiz.projects import registry


def test_list_shows_every_project(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    printed = capsys.readouterr().out
    for project in registry():
        assert project.name in printed
        assert project.title in printed


def test_single_project_is_rebuilt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "carsharing", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "отчёт.md" in printed

    folder = tmp_path / "03-тарифы-каршеринга"
    assert (folder / "отчёт.md").exists()
    assert (folder / "данные.csv").exists()
    assert list((folder / "графики").glob("*.svg"))


def test_seed_changes_the_data(tmp_path: Path) -> None:
    main(["run", "carsharing", "--out", str(tmp_path / "первый"), "--seed", "1"])
    main(["run", "carsharing", "--out", str(tmp_path / "второй"), "--seed", "2"])
    first = (tmp_path / "первый" / "03-тарифы-каршеринга" / "данные.csv").read_text("utf-8-sig")
    second = (tmp_path / "второй" / "03-тарифы-каршеринга" / "данные.csv").read_text("utf-8-sig")
    assert first != second


def test_unknown_project_is_explained(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "нет-такого", "--out", str(tmp_path)]) == 2
    printed = capsys.readouterr().err
    assert "нет-такого" in printed
    assert "carsharing" in printed, "в ошибке перечислены доступные исследования"


def test_all_projects_are_rebuilt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["all", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "Готово: 8 исследований" in printed
    for project in registry():
        assert (tmp_path / project.folder / "отчёт.md").exists()
