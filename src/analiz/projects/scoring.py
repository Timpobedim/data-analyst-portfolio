"""Исследование 1. Кто не возвращает рассрочку.

Классическая первая работа аналитика: заявки на рассрочку, грязные данные, вопрос «от чего зависит
просрочка». Интерес не в том, чтобы построить модель, а в том, чтобы отделить связи, которые видны
в данных, от тех, которые кажутся.

Отдельно проверяется то, о чём в учебных работах забывают: четыре гипотезы, проверенные с уровнем
0,05, дают примерно один ложный вывод из четырёх. Поэтому здесь применяется поправка Холма.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analiz import charts, data, quality, stats
from analiz.projects import Project
from analiz.report import Report, Section, number, percent

NAME: Final = "scoring"
TITLE: Final = "Кто не возвращает рассрочку"
FOLDER: Final = "01-скоринг-рассрочки"
QUESTION: Final = "Какие признаки заявителя действительно связаны с просрочкой по рассрочке?"

PURPOSES: Final = ("техника", "мебель", "ремонт", "образование", "лечение", "отпуск")
FAMILY: Final = ("женат/замужем", "не женат/не замужем", "в разводе", "вдовец/вдова")
APPLICATIONS: Final = 12_000


def generate(seed: int = data.SEED) -> pd.DataFrame:
    """Заявки на рассрочку: связь просрочки с детьми и целью заложена намеренно, остальное — шум."""
    rng = data.generator(seed)
    count = APPLICATIONS

    age = np.clip(rng.normal(38, 11, count), 19, 75).round(0)
    income = data.money(np.exp(rng.normal(10.9, 0.45, count)))
    experience = np.clip(age - 20 + rng.normal(0, 4, count), 0, 45).round(0)
    children = rng.choice([0, 1, 2, 3, 4], size=count, p=[0.48, 0.27, 0.17, 0.06, 0.02])
    purpose = rng.choice(PURPOSES, size=count, p=[0.28, 0.2, 0.18, 0.12, 0.12, 0.1])
    family = rng.choice(FAMILY, size=count, p=[0.55, 0.25, 0.14, 0.06])
    amount = data.money(np.clip(rng.normal(90_000, 45_000, count), 8_000, 400_000))
    months = rng.choice([3, 6, 10, 12, 24], size=count, p=[0.2, 0.3, 0.2, 0.2, 0.1])

    # Заложенная зависимость: дети и «отпуск» повышают риск, высокий доход понижает.
    risk = (
        0.06
        + 0.022 * children
        + 0.05 * (purpose == "отпуск")
        + 0.02 * (purpose == "лечение")
        - 0.03 * (income > 90_000)
        + 0.015 * (months >= 12)
    )
    overdue = rng.random(count) < np.clip(risk, 0.01, 0.6)

    frame = pd.DataFrame(
        {
            "возраст": age.astype(int),
            "доход": income,
            "стаж": experience.astype(int),
            "детей": children,
            "цель": purpose,
            "семейное_положение": family,
            "сумма": amount,
            "срок_месяцев": months,
            "просрочка": overdue.astype(int),
        }
    )

    frame = data.spoil_missing(frame, "доход", 0.07, rng)
    frame = data.spoil_missing(frame, "стаж", 0.04, rng)
    frame = data.spoil_outliers(frame, "доход", 0.004, 100.0, rng)
    return data.spoil_duplicates(frame, 0.01, rng)


def run(out: Path, seed: int = data.SEED) -> Report:
    raw = generate(seed)
    data.save_csv(raw, out / "данные.csv")
    pictures = out / "графики"

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"{len(raw)} заявок на рассрочку: возраст, доход, стаж, дети, цель, сумма и срок. "
            "Набор создаётся генератором с зафиксированным зерном и лежит рядом в «данные.csv»."
        ),
        method=(
            "подготовка данных, доли просрочки по группам, критерий хи-квадрат с размером эффекта "
            "(V Крамера) и поправка Холма на четыре проверки"
        ),
        prepared_on=datetime.now(UTC).date(),
    )

    checked = quality.inspect(raw)
    headers, rows = checked.table()
    report.add(
        Section(
            title="Что пришло",
            text=checked.summary() + " Пропуски в доходе и стаже — обычное дело: эти поля заполняет менеджер вручную.",
            headers=headers,
            rows=rows,
        )
    )

    prepared, cleaning = _prepare(raw)
    bounds = quality.outliers(prepared["доход"])
    report.add(
        Section(
            title="Что сделали с данными",
            text=(
                f"{cleaning.line()} Пропущенный доход заполнен медианой по возрастной группе, "
                f"а не общей медианой: у двадцатилетних и сорокалетних это разные деньги. "
                f"{bounds.line('доход')} — такие заявки убраны как ошибки ввода."
            ),
        )
    )

    overall = float(prepared["просрочка"].mean())
    hypotheses = _hypotheses(prepared)
    decisions = stats.holm([item.result.p_value for item in hypotheses])

    report.add(
        Section(
            title="Проверка четырёх гипотез",
            text=(
                f"Общая доля просрочки — {percent(overall)}. Каждая гипотеза проверена критерием "
                "хи-квадрат; поправка Холма нужна, потому что проверок несколько."
            ),
            headers=["Гипотеза", "p-значение", "V Крамера", "Значима", "С поправкой Холма"],
            rows=[
                [
                    item.title,
                    f"{item.result.p_value:.4f}",
                    f"{item.result.effect:.3f}" if item.result.effect is not None else "—",
                    "да" if item.result.significant else "нет",
                    "да" if decision else "нет",
                ]
                for item, decision in zip(hypotheses, decisions, strict=True)
            ],
        )
    )

    by_children = prepared.groupby("детей")["просрочка"].mean()
    report.add(
        Section(
            title="Дети",
            text="Доля просрочки растёт вместе с числом детей — это самая заметная связь в наборе.",
            image=charts.bars(
                [str(index) for index in by_children.index],
                [float(value) for value in by_children],
                path=pictures / "просрочка-по-детям.svg",
                title="Доля просрочки по числу детей",
                subtitle="доля заявок с просрочкой",
                ylabel="доля",
            ),
        )
    )

    by_purpose = prepared.groupby("цель")["просрочка"].mean().sort_values(ascending=False)
    report.add(
        Section(
            title="Цель рассрочки",
            text="Рассрочка на отпуск заканчивается просрочкой заметно чаще, чем на технику или мебель.",
            headers=["Цель", "Заявок", "Доля просрочки"],
            rows=[
                [
                    str(purpose),
                    number(float((prepared["цель"] == purpose).sum())),
                    percent(float(value)),
                ]
                for purpose, value in by_purpose.items()
            ],
            image=charts.bars(
                [str(index) for index in by_purpose.index],
                [float(value) for value in by_purpose],
                path=pictures / "просрочка-по-целям.svg",
                title="Доля просрочки по цели рассрочки",
                subtitle="отсортировано по убыванию",
                ylabel="доля",
                horizontal=True,
            ),
        )
    )

    rich = prepared[prepared["доход"] > prepared["доход"].median()]["просрочка"]
    poor = prepared[prepared["доход"] <= prepared["доход"].median()]["просрочка"]
    income_test = stats.compare_shares(
        int(rich.sum()), len(rich), int(poor.sum()), len(poor), labels=("доход выше медианы", "доход ниже медианы")
    )
    report.add(
        Section(
            title="Доход",
            text=income_test.conclusion,
            image=charts.histogram(
                [float(value) for value in prepared["доход"]],
                path=pictures / "доход.svg",
                title="Распределение дохода заявителей",
                subtitle="после удаления ошибок ввода",
                xlabel="доход, ₽",
            ),
        )
    )

    significant = [item.title for item, decision in zip(hypotheses, decisions, strict=True) if decision]
    report.findings = [
        f"Общая доля просрочки — {percent(overall)}; это опорное число для любых сравнений.",
        f"После поправки на множественные сравнения значимыми остаются: {', '.join(significant)}."
        if significant
        else "После поправки на множественные сравнения ни одна гипотеза не устояла.",
        f"Заявители с доходом выше медианы допускают просрочку реже: {income_test.conclusion.lower()}",
        "Размеры эффекта малы (V Крамера меньше 0,1): признаки связаны с просрочкой, но по одному "
        "из них решение принимать нельзя.",
    ]
    report.limits = [
        "Данные синтетические: зависимости в них заложены генератором, поэтому выводы показывают метод, а не рынок.",
        "Связь не означает причину: у заявителей с детьми выше не «риск как таковой», а расходы.",
        "Скоринговой модели здесь нет: это разведочный анализ, а не оценка кредитоспособности.",
        "Отказы в рассрочке в данные не попали, поэтому выборка смещена: мы видим только одобренные заявки.",
    ]
    return report


@dataclass(frozen=True, slots=True)
class Hypothesis:
    title: str
    result: stats.TestResult


def _prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, quality.CleanReport]:
    frame = raw.copy()
    frame["возрастная_группа"] = pd.cut(
        frame["возраст"], bins=[18, 25, 35, 45, 60, 100], labels=["19–25", "26–35", "36–45", "46–60", "60+"]
    ).astype(str)
    # Медиана по возрастной группе: доход двадцатилетнего и сорокалетнего — разные величины.
    frame["доход"] = frame.groupby("возрастная_группа")["доход"].transform(
        lambda values: values.fillna(values.median())
    )
    frame["стаж"] = frame["стаж"].fillna(frame["стаж"].median())
    return quality.clean(frame, required=["доход", "стаж"], trim=["доход"])


def _hypotheses(frame: pd.DataFrame) -> list[Hypothesis]:
    return [
        Hypothesis(
            "Просрочка связана с числом детей", stats.chi_square(_table(frame, "детей"), what="детьми и просрочкой")
        ),
        Hypothesis(
            "Просрочка связана с целью рассрочки", stats.chi_square(_table(frame, "цель"), what="целью и просрочкой")
        ),
        Hypothesis(
            "Просрочка связана с семейным положением",
            stats.chi_square(_table(frame, "семейное_положение"), what="семейным положением и просрочкой"),
        ),
        Hypothesis(
            "Просрочка связана со сроком рассрочки",
            stats.chi_square(_table(frame, "срок_месяцев"), what="сроком и просрочкой"),
        ),
    ]


def _table(frame: pd.DataFrame, column: str) -> list[list[int]]:
    counted = pd.crosstab(frame[column], frame["просрочка"])
    return [[int(value) for value in row] for row in counted.to_numpy()]


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
