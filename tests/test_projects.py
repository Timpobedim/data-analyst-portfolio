from __future__ import annotations

from pathlib import Path

import pytest

from analiz.projects import Project, find, registry

FAST = "carsharing"


def test_registry_holds_eight_unique_projects() -> None:
    projects = registry()
    assert len(projects) == 8
    assert len({project.name for project in projects}) == 8
    assert len({project.folder for project in projects}) == 8
    assert all(project.question.endswith("?") for project in projects), "у каждого исследования есть вопрос"


def test_find_knows_its_projects() -> None:
    assert find(FAST) is not None
    assert find("нет-такого") is None


@pytest.mark.parametrize("project", registry(), ids=lambda item: item.name)
def test_every_project_builds_a_full_report(project: Project, tmp_path: Path) -> None:
    folder = tmp_path / project.folder
    report = project.run(folder, 42)

    assert report.title == project.title
    assert report.question == project.question
    assert len(report.sections) >= 4, "отчёт без разделов не отчёт"
    assert len(report.findings) >= 3
    assert len(report.limits) >= 3, "раздел «чего это не доказывает» обязателен"

    text = report.to_markdown()
    assert "## Что получилось" in text
    assert "## Чего это не доказывает" in text
    assert len(list(folder.glob("*.csv"))) >= 1, "данные исследования лежат рядом с отчётом"
    assert len(list((folder / "графики").glob("*.svg"))) >= 2


def test_same_seed_gives_the_same_report(tmp_path: Path) -> None:
    project = find(FAST)
    assert project is not None
    first = project.run(tmp_path / "первый", 42).to_markdown()
    second = project.run(tmp_path / "второй", 42).to_markdown()
    assert first == second, "исследование должно быть воспроизводимым"

    other = project.run(tmp_path / "третий", 7).to_markdown()
    assert other != first, "другое зерно — другие данные"


def test_ab_test_shows_the_peeking_problem(tmp_path: Path) -> None:
    project = find("delivery_ab")
    assert project is not None
    text = project.run(tmp_path / "ab", 42).to_markdown()
    assert "подглядывать" in text
    assert "мощностью 80%" in text, "размер выборки считается до выводов"
    assert "доверительный интервал" in text.lower()


def test_churn_explains_the_threshold(tmp_path: Path) -> None:
    project = find("churn")
    assert project is not None
    report = project.run(tmp_path / "churn", 42)
    text = report.to_markdown()
    assert "порог" in text.lower()
    assert "утечк" in text.lower(), "признак-подсказка должен быть назван прямо"
    assert "ROC-AUC" in text


def test_sql_project_shows_its_queries(tmp_path: Path) -> None:
    project = find("bookstore")
    assert project is not None
    text = project.run(tmp_path / "sql", 42).to_markdown()
    assert "```sql" in text
    assert "OVER (" in text
    assert "rank()" in text.lower()
