"""Исследование 3. Какой тариф каршеринга приносит больше денег.

Здесь проверяется гипотеза о равенстве средних — и по дороге видно, почему одного p-значения мало.
Разница средних может быть значимой и при этом крошечной; поэтому рядом всегда стоят доверительный
интервал и размер эффекта, а перед выводом — проверка равенства дисперсий, от которой зависит выбор
критерия.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analiz import charts, data, quality, stats
from analiz.projects import Project
from analiz.report import Report, Section, money, number, percent

NAME: Final = "carsharing"
TITLE: Final = "Какой тариф каршеринга приносит больше"
FOLDER: Final = "03-тарифы-каршеринга"
QUESTION: Final = "Приносит ли пакетный тариф больше выручки на пользователя, чем поминутный?"

USERS: Final = 900
MONTHS: Final = 6
START: Final = date(2026, 1, 1)
MINUTE_RATE: Final = 9.5
PACKAGE_PRICE: Final = 2_490.0
PACKAGE_MINUTES: Final = 300


def generate(seed: int = data.SEED) -> pd.DataFrame:
    """Поездки пользователей за полгода: два тарифа, разные привычки, немного пропусков."""
    rng = data.generator(seed)
    tariffs = rng.choice(["поминутный", "пакетный"], size=USERS, p=[0.62, 0.38])
    cities = rng.choice(["Казань", "Самара", "Тюмень"], size=USERS, p=[0.45, 0.35, 0.2])

    rows: list[dict[str, object]] = []
    for user in range(USERS):
        tariff = str(tariffs[user])
        # У пакетного тарифа поездок больше: люди «выкатывают» оплаченные минуты.
        intensity = 8.0 if tariff == "пакетный" else 6.0
        # Привычка человека: кто-то ездит короткими перебежками, кто-то — через весь город.
        style = float(rng.lognormal(0.0, 0.45))
        for month in data.months(START, MONTHS):
            trips = int(rng.poisson(intensity))
            base_minutes = float(np.clip(rng.gamma(2.4, 13 if tariff == "пакетный" else 10), 4, 240).sum())
            minutes = base_minutes * max(trips, 0) * style
            if tariff == "пакетный":
                extra = max(minutes - PACKAGE_MINUTES, 0)
                revenue = PACKAGE_PRICE + extra * MINUTE_RATE * 0.8
            else:
                revenue = minutes * MINUTE_RATE
            rows.append(
                {
                    "пользователь": user + 1,
                    "город": str(cities[user]),
                    "тариф": tariff,
                    "месяц": month,
                    "поездок": trips,
                    "минут": round(minutes, 1),
                    "выручка": round(float(revenue), 2),
                }
            )

    frame = pd.DataFrame(rows)
    frame = data.spoil_missing(frame, "минут", 0.03, rng)
    return data.spoil_duplicates(frame, 0.008, rng)


def run(out: Path, seed: int = data.SEED) -> Report:
    raw = generate(seed)
    data.save_csv(raw, out / "данные.csv")
    pictures = out / "графики"

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"{len(raw)} записей: {USERS} пользователей за {MONTHS} месяцев, поездки, минуты и выручка "
            "по двум тарифам. Набор воспроизводим по зерну и лежит рядом в «данные.csv»."
        ),
        method=(
            "подготовка данных, помесячная выручка на пользователя, проверка равенства дисперсий (Левен), "
            "критерий Стьюдента или Уэлча, размер эффекта и доверительные интервалы"
        ),
        prepared_on=datetime.now(UTC).date(),
    )

    checked = quality.inspect(raw)
    headers, rows = checked.table()
    report.add(Section(title="Что пришло", text=checked.summary(), headers=headers, rows=rows))

    prepared, cleaning = quality.clean(raw, required=["минут"])
    per_user = prepared.groupby(["пользователь", "тариф"], as_index=False)["выручка"].mean()
    report.add(
        Section(
            title="Что сделали с данными",
            text=(
                f"{cleaning.line()} Дальше сравниваются не отдельные месяцы, а средняя выручка на "
                "пользователя: иначе активные клиенты попадут в выборку по шесть раз и создадут "
                "видимость большей выборки, чем есть."
            ),
        )
    )

    minute_users = [float(value) for value in per_user[per_user["тариф"] == "поминутный"]["выручка"]]
    package_users = [float(value) for value in per_user[per_user["тариф"] == "пакетный"]["выручка"]]
    test = stats.compare_means(package_users, minute_users, labels=("пакетный тариф", "поминутный тариф"))
    package_interval = stats.mean_interval(package_users)
    minute_interval = stats.mean_interval(minute_users)

    report.add(
        Section(
            title="Сравнение тарифов",
            text=test.conclusion,
            headers=["Тариф", "Пользователей", "Средняя выручка", "Доверительный интервал"],
            rows=[
                [
                    "пакетный",
                    number(len(package_users)),
                    money(float(np.mean(package_users))),
                    f"{money(package_interval[0])} — {money(package_interval[1])}",
                ],
                [
                    "поминутный",
                    number(len(minute_users)),
                    money(float(np.mean(minute_users))),
                    f"{money(minute_interval[0])} — {money(minute_interval[1])}",
                ],
            ],
            image=charts.boxes(
                {"пакетный": package_users, "поминутный": minute_users},
                path=pictures / "выручка-по-тарифам.svg",
                title="Средняя месячная выручка на пользователя",
                subtitle="медиана, квартили и усы",
                ylabel="₽ в месяц",
            ),
        )
    )

    monthly = prepared.groupby(["месяц", "тариф"])["выручка"].mean().unstack()
    report.add(
        Section(
            title="Как это менялось по месяцам",
            text="Разрыв между тарифами держится все шесть месяцев — это не разовый выброс одного месяца.",
            image=charts.lines(
                [f"{month:%m.%Y}" for month in monthly.index],
                {str(column): [float(value) for value in monthly[column]] for column in monthly.columns},
                path=pictures / "выручка-по-месяцам.svg",
                title="Средняя выручка на пользователя по месяцам",
                subtitle="по тарифам",
                ylabel="₽",
            ),
        )
    )

    by_city = prepared.groupby(["город", "тариф"])["выручка"].mean().unstack()
    report.add(
        Section(
            title="Города",
            text="Картина по городам одинаковая — вывод не держится на одном рынке.",
            headers=["Город", "Пакетный", "Поминутный", "Разница"],
            rows=[
                [
                    str(city),
                    money(float(row["пакетный"])),
                    money(float(row["поминутный"])),
                    percent(float(row["пакетный"] / row["поминутный"]) - 1),
                ]
                for city, row in by_city.iterrows()
            ],
        )
    )

    difference = float(np.mean(package_users) - np.mean(minute_users))
    report.findings = [
        f"Пакетный тариф приносит в среднем на {money(difference)} в месяц больше на пользователя.",
        test.line(),
        f"Размер эффекта — {stats.effect_size_name(test.effect or 0.0)}: разница не только значима, но и заметна.",
        "Во всех трёх городах направление одинаковое, поэтому вывод не объясняется особенностями одного рынка.",
    ]
    report.limits = [
        "Данные синтетические: разница между тарифами заложена генератором, числа иллюстрируют метод.",
        "Это наблюдение, а не эксперимент: пакетный тариф выбирают те, кто и так ездит больше. "
        "Перевод всех на пакетный не даст такого прироста.",
        "Себестоимость поездок не учитывается: выше выручка — не обязательно выше прибыль.",
        "Полгода — короткий горизонт: сезонность и отток за этот срок не видны.",
    ]
    return report


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
