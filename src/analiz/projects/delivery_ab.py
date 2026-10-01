"""Исследование 5. A/B-тест баннера в приложении доставки.

Три вещи, которые превращают A/B-тест из ритуала в инструмент, и которые здесь показаны явно.

Первое: размер выборки считается **до** запуска. Иначе тест заканчивается фразой «различий не
обнаружено», которая ничего не означает — мощности могло не хватить с самого начала.

Второе: подглядывание. Если смотреть на p-значение каждый день и останавливаться, как только оно
опустилось ниже 0,05, доля ложных срабатываний растёт в разы. Здесь p-значение по дням посчитано
и показано отдельно — видно, как оно ныряет и возвращается.

Третье: значимость — не то же самое, что польза. Рядом с p-значением всегда стоит доверительный
интервал разницы и пересчёт в деньги.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analiz import charts, data, quality, stats
from analiz.projects import Project
from analiz.report import Report, Section, change, money, number, percent

NAME: Final = "delivery_ab"
TITLE: Final = "A/B-тест баннера в приложении доставки"
FOLDER: Final = "05-ab-тест-баннера"
QUESTION: Final = "Увеличил ли новый баннер долю заказов и стоит ли его раскатывать на всех?"

USERS: Final = 24_000
DAYS: Final = 14
START: Final = date(2026, 4, 1)
BASE_CONVERSION: Final = 0.086
REAL_LIFT: Final = 0.009
CITIES: Final = ("Москва", "Санкт-Петербург", "Новосибирск", "Краснодар")
PLATFORMS: Final = ("Android", "iOS")


def generate(seed: int = data.SEED) -> pd.DataFrame:
    """Пользователи эксперимента: группа, город, платформа, заказ и чек."""
    rng = data.generator(seed)
    group = rng.choice(["контроль", "баннер"], size=USERS, p=[0.5, 0.5])
    city = rng.choice(CITIES, size=USERS, p=[0.4, 0.25, 0.2, 0.15])
    platform = rng.choice(PLATFORMS, size=USERS, p=[0.62, 0.38])
    day = rng.integers(0, DAYS, size=USERS)

    probability = np.where(group == "баннер", BASE_CONVERSION + REAL_LIFT, BASE_CONVERSION)
    ordered = rng.random(USERS) < probability
    # Чек скошен вправо: несколько крупных заказов тянут среднее.
    receipt = np.where(ordered, data.money(np.exp(rng.normal(6.9, 0.5, USERS))), 0.0)

    frame = pd.DataFrame(
        {
            "пользователь": np.arange(1, USERS + 1),
            "группа": group,
            "город": city,
            "платформа": platform,
            "день": [START + pd.Timedelta(days=int(value)) for value in day],
            "заказ": ordered.astype(int),
            "чек": receipt,
        }
    )
    return data.spoil_duplicates(frame, 0.004, rng)


def run(out: Path, seed: int = data.SEED) -> Report:
    raw = generate(seed)
    data.save_csv(raw, out / "данные.csv")
    pictures = out / "графики"

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"{len(raw)} назначений пользователей в эксперимент за {DAYS} дней: группа, город, платформа, "
            "факт заказа и сумма чека. Набор воспроизводим по зерну."
        ),
        method=(
            "расчёт нужного размера выборки до теста, проверка сбалансированности групп, z-критерий для "
            "долей с доверительным интервалом, критерий Манна–Уитни для чека и проверка на подглядывание"
        ),
        prepared_on=datetime.now(UTC).date(),
    )

    checked = quality.inspect(raw)
    headers, rows = checked.table()
    report.add(Section(title="Что пришло", text=checked.summary(), headers=headers, rows=rows))

    prepared, cleaning = quality.clean(raw)
    needed = stats.required_sample(BASE_CONVERSION, REAL_LIFT)
    per_group = int(prepared.groupby("группа").size().min())
    enough = per_group >= needed
    verdict = "этого достаточно" if enough else "этого мало, и отрицательный результат ничего не докажет"
    report.add(
        Section(
            title="Хватит ли данных: расчёт до выводов",
            text=(
                f"{cleaning.line()} Чтобы заметить прирост конверсии на {percent(REAL_LIFT)} при базовой "
                f"{percent(BASE_CONVERSION)} с уровнем 0,05 и мощностью 80%, нужно {number(needed)} "
                f"пользователей в каждой группе. Фактически в меньшей группе {number(per_group)} — "
                f"{verdict}."
            ),
        )
    )

    balance = pd.crosstab(prepared["группа"], prepared["город"])
    balance_test = stats.chi_square(
        [[int(value) for value in row] for row in balance.to_numpy()], what="группой и городом"
    )
    report.add(
        Section(
            title="Сбалансированы ли группы",
            text=(
                "До сравнения результатов проверяется состав групп: если в одной больше москвичей, "
                "разница в конверсии может объясняться этим, а не баннером. " + balance_test.conclusion
            ),
            headers=["Группа", *[str(city) for city in balance.columns], "Всего"],
            rows=[
                [str(name), *[number(float(value)) for value in row], number(float(row.sum()))]
                for name, row in balance.iterrows()
            ],
        )
    )

    control = prepared[prepared["группа"] == "контроль"]
    banner = prepared[prepared["группа"] == "баннер"]
    conversion_test = stats.compare_shares(
        int(banner["заказ"].sum()),
        len(banner),
        int(control["заказ"].sum()),
        len(control),
        labels=("группа с баннером", "контроль"),
    )
    banner_share = float(banner["заказ"].mean())
    control_share = float(control["заказ"].mean())

    report.add(
        Section(
            title="Конверсия в заказ",
            text=conversion_test.conclusion,
            headers=["Группа", "Пользователей", "Заказов", "Конверсия", "Доверительный интервал"],
            rows=[
                [
                    "баннер",
                    number(len(banner)),
                    number(float(banner["заказ"].sum())),
                    percent(banner_share),
                    _interval(int(banner["заказ"].sum()), len(banner)),
                ],
                [
                    "контроль",
                    number(len(control)),
                    number(float(control["заказ"].sum())),
                    percent(control_share),
                    _interval(int(control["заказ"].sum()), len(control)),
                ],
            ],
            image=charts.bars(
                ["контроль", "баннер"],
                [control_share, banner_share],
                path=pictures / "конверсия.svg",
                title="Конверсия в заказ по группам",
                subtitle="доля пользователей, сделавших заказ",
                ylabel="доля",
            ),
        )
    )

    daily = _daily_p_values(prepared)
    dips = sum(1 for value in daily["p"][:-1] if value < stats.ALPHA)
    report.add(
        Section(
            title="Почему нельзя подглядывать",
            text=(
                "Если бы решение принимали в первый же день, когда p-значение опустилось ниже 0,05, "
                f"тест остановили бы раньше срока: за {DAYS} дней это случалось {dips} раз "
                "до окончания эксперимента. Итоговый вывод делается один раз — по всей выборке."
            ),
            image=charts.lines(
                [f"{day:%d.%m}" for day in daily["дни"]],
                {"p-значение по накопленным данным": daily["p"], "уровень 0,05": [stats.ALPHA] * len(daily["p"])},
                path=pictures / "подглядывание.svg",
                title="p-значение, если смотреть каждый день",
                subtitle="накопленные данные с начала теста",
                ylabel="p-значение",
            ),
        )
    )

    banner_checks = [float(value) for value in banner[banner["заказ"] == 1]["чек"]]
    control_checks = [float(value) for value in control[control["заказ"] == 1]["чек"]]
    receipt_test = stats.compare_distributions(
        banner_checks, control_checks, labels=("чек с баннером", "чек в контроле")
    )
    report.add(
        Section(
            title="Средний чек",
            text=(
                "Чек сильно скошен вправо, поэтому сравниваются распределения, а не средние. " + receipt_test.conclusion
            ),
            image=charts.histogram(
                banner_checks + control_checks,
                path=pictures / "чек.svg",
                title="Распределение чека оформленных заказов",
                subtitle="обе группы вместе",
                bins=40,
                xlabel="чек, ₽",
            ),
        )
    )

    lift = banner_share - control_share
    monthly_users = 300_000
    extra_orders = monthly_users * lift
    average_check = float(np.mean(banner_checks + control_checks))
    report.findings = [
        conversion_test.line(),
        f"Прирост конверсии — {change(lift / control_share)} относительно контроля "
        f"({percent(control_share)} → {percent(banner_share)}).",
        f"При {number(monthly_users)} пользователей в месяц это примерно {number(extra_orders)} "
        f"дополнительных заказов и {money(extra_orders * average_check)} выручки.",
        receipt_test.conclusion,
    ]
    report.limits = [
        "Данные синтетические: прирост заложен генератором, числа показывают метод проверки, а не эффект настоящего "
        "баннера.",
        "Две недели — короткий срок: эффект новизны за это время не успевает угаснуть, и на длинном горизонте прирост "
        "может оказаться меньше.",
        "Проверялась одна метрика-цель; влияние на возвраты, отписки и нагрузку на поддержку не измерялось.",
        "Пользователи распределены случайно один раз; повторные визиты одного человека в разных группах не "
        "моделируются.",
    ]
    return report


def _interval(successes: int, total: int) -> str:
    low, high = stats.share_interval(successes, total)
    return f"{percent(low)} — {percent(high)}"


def _daily_p_values(frame: pd.DataFrame) -> dict[str, list[float]]:
    """p-значение по накопленным данным на каждый день теста."""
    days = sorted(frame["день"].unique())
    values: list[float] = []
    for day in days:
        window = frame[frame["день"] <= day]
        banner = window[window["группа"] == "баннер"]
        control = window[window["группа"] == "контроль"]
        if len(banner) < 2 or len(control) < 2:
            values.append(1.0)
            continue
        result = stats.compare_shares(
            int(banner["заказ"].sum()), len(banner), int(control["заказ"].sum()), len(control)
        )
        values.append(result.p_value)
    return {"дни": days, "p": values}


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
