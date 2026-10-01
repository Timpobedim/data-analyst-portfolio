"""Исследование 4. Какие настольные игры покупают.

Исследование рынка: жанры, издатели, оценки и продажи. Главный методический вопрос здесь — какой
период брать. Данные за пятнадцать лет отвечают на вопрос «что покупали», а решение принимается про
«что покупают»: поэтому актуальный срез определяется явно и обосновывается, а не берётся молча.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from analiz import charts, data, quality, stats
from analiz.projects import Project
from analiz.report import Report, Section, money, number, percent

NAME: Final = "boardgames"
TITLE: Final = "Какие настольные игры покупают"
FOLDER: Final = "04-рынок-настольных-игр"
QUESTION: Final = "Какие жанры настольных игр растут и стоит ли ориентироваться на оценки?"

GENRES: Final = {
    "семейные": 1.25,
    "стратегии": 1.0,
    "детективы": 0.85,
    "детские": 1.1,
    "party": 0.95,
    "варгеймы": 0.45,
}
PUBLISHERS: Final = ("Хобби Мир", "Мосигра", "Стиль Жизни", "Звезда", "Правильные игры", "GaGa")
GAMES: Final = 3_500
FIRST_YEAR: Final = 2011
LAST_YEAR: Final = 2025
RECENT_YEARS: Final = 4


def generate(seed: int = data.SEED) -> pd.DataFrame:
    """Каталог изданных игр: продажи растут у семейных и детских, падают у варгеймов."""
    rng = data.generator(seed)
    count = GAMES
    years = rng.integers(FIRST_YEAR, LAST_YEAR + 1, size=count)
    genre_names = list(GENRES)
    genre = rng.choice(genre_names, size=count, p=[0.26, 0.22, 0.14, 0.18, 0.14, 0.06])
    publisher = rng.choice(PUBLISHERS, size=count)

    critics = np.clip(rng.normal(7.1, 1.2, count), 1, 10).round(1)
    players = np.clip(critics + rng.normal(-0.3, 1.0, count), 1, 10).round(1)
    price = data.money(np.clip(rng.normal(2_800, 1_200, count), 490, 12_000))

    trend = np.array([GENRES[name] for name in genre])
    age = (years - FIRST_YEAR) / (LAST_YEAR - FIRST_YEAR)
    growth = np.where(np.isin(genre, ["семейные", "детские"]), 1 + 0.9 * age, 1 - 0.45 * age)
    base = rng.gamma(1.7, 2_600, count) * trend * growth
    quality_bonus = 1 + 0.11 * (players - players.mean())
    sales = np.round(np.clip(base * quality_bonus, 100, None)).astype(int)

    frame = pd.DataFrame(
        {
            "название": [f"Игра №{number + 1}" for number in range(count)],
            "год": years,
            "жанр": genre,
            "издатель": publisher,
            "оценка_критиков": critics,
            "оценка_игроков": players,
            "цена": price,
            "продано_копий": sales,
        }
    )

    frame = data.spoil_missing(frame, "оценка_критиков", 0.18, rng)
    frame = data.spoil_missing(frame, "оценка_игроков", 0.09, rng)
    return data.spoil_duplicates(frame, 0.006, rng)


def run(out: Path, seed: int = data.SEED) -> Report:
    raw = generate(seed)
    data.save_csv(raw, out / "данные.csv")
    pictures = out / "графики"

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"{len(raw)} изданий настольных игр за {FIRST_YEAR}–{LAST_YEAR} годы: жанр, издатель, "
            "оценки критиков и игроков, цена и тираж. Набор воспроизводим по зерну."
        ),
        method=(
            "подготовка данных, выбор актуального периода, доли жанров по годам, корреляция Спирмена "
            "между оценками и продажами, критерий Манна–Уитни для сравнения оценок"
        ),
        prepared_on=datetime.now(UTC).date(),
    )

    checked = quality.inspect(raw)
    headers, rows = checked.table()
    report.add(
        Section(
            title="Что пришло",
            text=(
                checked.summary() + " Оценки критиков есть не у всех игр: небольшие издания просто не рецензируют. "
                "Такие пропуски нельзя заполнять средним — это придумало бы рецензии, которых не было."
            ),
            headers=headers,
            rows=rows,
        )
    )

    prepared, cleaning = quality.clean(raw)
    recent = prepared[prepared["год"] > LAST_YEAR - RECENT_YEARS]
    report.add(
        Section(
            title="Какой период считать актуальным",
            text=(
                f"{cleaning.line()} Для выводов о сегодняшнем рынке взяты последние {RECENT_YEARS} года "
                f"({LAST_YEAR - RECENT_YEARS + 1}–{LAST_YEAR}): в них {number(len(recent))} изданий "
                f"({percent(len(recent) / len(prepared))} каталога). Более ранние годы оставлены "
                "для сравнения динамики, но не участвуют в выводах о текущих предпочтениях."
            ),
        )
    )

    by_year = prepared.pivot_table(index="год", columns="жанр", values="продано_копий", aggfunc="sum").fillna(0)
    report.add(
        Section(
            title="Динамика жанров",
            text="Семейные и детские игры растут, варгеймы сжимаются — это видно на всём горизонте.",
            image=charts.lines(
                [str(year) for year in by_year.index],
                {str(column): [float(value) for value in by_year[column]] for column in by_year.columns},
                path=pictures / "жанры-по-годам.svg",
                title="Продажи по жанрам",
                subtitle="суммарный тираж за год, копий",
                ylabel="копий",
            ),
        )
    )

    genre_recent = (
        recent.groupby("жанр")
        .agg(изданий=("название", "size"), тираж=("продано_копий", "sum"), медиана=("продано_копий", "median"))
        .sort_values("тираж", ascending=False)
    )
    total_recent = float(genre_recent["тираж"].sum())
    report.add(
        Section(
            title=f"Жанры за {LAST_YEAR - RECENT_YEARS + 1}–{LAST_YEAR}",
            headers=["Жанр", "Изданий", "Тираж", "Доля рынка", "Медианный тираж"],
            rows=[
                [
                    str(name),
                    number(float(row["изданий"])),
                    number(float(row["тираж"])),
                    percent(float(row["тираж"]) / total_recent),
                    number(float(row["медиана"])),
                ]
                for name, row in genre_recent.iterrows()
            ],
            image=charts.bars(
                [str(name) for name in genre_recent.index],
                [float(value) for value in genre_recent["тираж"]],
                path=pictures / "жанры-актуальные.svg",
                title=f"Тираж по жанрам за {LAST_YEAR - RECENT_YEARS + 1}–{LAST_YEAR}",
                subtitle="суммарно, копий",
                ylabel="копий",
                horizontal=True,
            ),
        )
    )

    publishers = (
        recent.groupby("издатель")
        .agg(изданий=("название", "size"), тираж=("продано_копий", "sum"), средняя_цена=("цена", "mean"))
        .sort_values("тираж", ascending=False)
    )
    report.add(
        Section(
            title="Издатели",
            headers=["Издатель", "Изданий", "Тираж", "Средняя цена"],
            rows=[
                [
                    str(name),
                    number(float(row["изданий"])),
                    number(float(row["тираж"])),
                    money(float(row["средняя_цена"])),
                ]
                for name, row in publishers.iterrows()
            ],
        )
    )

    with_scores = recent.dropna(subset=["оценка_критиков", "оценка_игроков"])
    critics_correlation = float(
        scipy_stats.spearmanr(with_scores["оценка_критиков"], with_scores["продано_копий"]).statistic
    )
    players_correlation = float(
        scipy_stats.spearmanr(with_scores["оценка_игроков"], with_scores["продано_копий"]).statistic
    )
    score_test = stats.compare_distributions(
        [float(value) for value in with_scores["оценка_критиков"]],
        [float(value) for value in with_scores["оценка_игроков"]],
        labels=("оценки критиков", "оценки игроков"),
    )
    report.add(
        Section(
            title="Оценки и продажи",
            text=(
                f"Связь оценки игроков с тиражом — {players_correlation:.2f} по Спирмену, оценки критиков — "
                f"{critics_correlation:.2f}. Обе связи слабые: оценка не предсказывает тираж. " + score_test.conclusion
            ),
            image=charts.scatter(
                [float(value) for value in with_scores["оценка_игроков"]],
                [float(value) for value in with_scores["продано_копий"]],
                path=pictures / "оценки-и-продажи.svg",
                title="Оценка игроков и тираж",
                subtitle=f"игры {LAST_YEAR - RECENT_YEARS + 1}–{LAST_YEAR} годов",
                xlabel="оценка игроков",
                ylabel="копий",
            ),
        )
    )

    leader = str(genre_recent.index[0])
    report.findings = [
        f"Самый крупный жанр последних лет — «{leader}»: "
        f"{percent(float(genre_recent['тираж'].iloc[0]) / total_recent)} тиража.",
        "Семейные и детские игры растут весь период, варгеймы сокращаются примерно вдвое.",
        f"Оценки слабо связаны с тиражом (Спирмен {players_correlation:.2f}): ставка на «высокий рейтинг» не окупается "
        "сама по себе.",
        score_test.conclusion,
    ]
    report.limits = [
        "Данные синтетические: тренды заложены генератором, поэтому выводы показывают метод, а не настоящий рынок.",
        "Тираж — не выручка: цена и скидки здесь не учитываются в расчёте дохода.",
        "Пропуски в оценках не случайны: у мелких изданий рецензий нет, и это смещает любую связь с оценками.",
        "Четыре года выбраны как компромисс между свежестью и объёмом; при другом окне выводы о динамике изменятся.",
    ]
    return report


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
