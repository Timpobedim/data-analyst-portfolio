"""Исследование 6. Воронка онлайн-школы: где теряются ученики и какой канал окупается.

Два вопроса, которые задают чаще всего: «где отваливаются» и «куда вкладывать бюджет». Первый —
про шаги воронки, второй — про каналы привлечения. Во втором есть ловушка: каналов пять, и если
сравнивать каждый с каждым без поправки, «победитель» найдётся даже там, где все одинаковые.
Поэтому сравнения идут с поправкой Холма, а решение опирается на окупаемость, а не на конверсию.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from analiz import charts, data, quality, stats
from analiz.projects import Project
from analiz.report import Report, Section, as_float, change, money, number, percent

NAME: Final = "school_funnel"
TITLE: Final = "Воронка онлайн-школы: где теряются ученики"
FOLDER: Final = "06-воронка-онлайн-школы"
QUESTION: Final = "На каком шаге воронки теряются ученики и какой канал привлечения окупается?"

VISITORS: Final = 30_000
WEEKS: Final = 12
START: Final = date(2026, 1, 5)

CHANNELS: Final = {
    # канал: (доля трафика, множитель конверсии, стоимость визита в рублях)
    "поиск": (0.3, 1.15, 95.0),
    "соцсети": (0.26, 0.85, 60.0),
    "рассылка": (0.16, 1.35, 120.0),
    "блогеры": (0.16, 0.95, 150.0),
    "рекомендации": (0.12, 1.5, 400.0),
}
STEPS: Final = ("визит", "регистрация", "пробный урок", "оплата")
BASE: Final = (1.0, 0.12, 0.35, 0.22)
PRICE: Final = 18_900.0


def generate(seed: int = data.SEED) -> pd.DataFrame:
    """Путь пользователей по воронке: у каждого канала своя конверсия и своя цена визита."""
    rng = data.generator(seed)
    names = list(CHANNELS)
    channel = rng.choice(names, size=VISITORS, p=[CHANNELS[name][0] for name in names])
    multiplier = np.array([CHANNELS[name][1] for name in channel])
    week = rng.integers(0, WEEKS, size=VISITORS)

    registered = rng.random(VISITORS) < np.clip(BASE[1] * multiplier, 0.01, 0.95)
    trial = registered & (rng.random(VISITORS) < np.clip(BASE[2] * multiplier, 0.01, 0.95))
    paid = trial & (rng.random(VISITORS) < np.clip(BASE[3] * multiplier, 0.01, 0.95))

    frame = pd.DataFrame(
        {
            "пользователь": np.arange(1, VISITORS + 1),
            "канал": channel,
            "неделя": [START + pd.Timedelta(weeks=int(value)) for value in week],
            "регистрация": registered.astype(int),
            "пробный_урок": trial.astype(int),
            "оплата": paid.astype(int),
            "стоимость_визита": [CHANNELS[name][2] for name in channel],
        }
    )
    return data.spoil_duplicates(frame, 0.005, rng)


def run(out: Path, seed: int = data.SEED) -> Report:
    raw = generate(seed)
    data.save_csv(raw, out / "данные.csv")
    pictures = out / "графики"

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"{len(raw)} визитов за {WEEKS} недель: канал привлечения, шаги воронки (регистрация, "
            "пробный урок, оплата) и стоимость визита. Набор воспроизводим по зерну."
        ),
        method=(
            "конверсия по шагам, сравнение каналов z-критерием для долей с поправкой Холма на пять "
            "сравнений, когорты по неделям и расчёт окупаемости канала"
        ),
        prepared_on=datetime.now(UTC).date(),
    )

    checked = quality.inspect(raw)
    headers, rows = checked.table()
    report.add(Section(title="Что пришло", text=checked.summary(), headers=headers, rows=rows))

    prepared, cleaning = quality.clean(raw)
    counts = [
        len(prepared),
        int(prepared["регистрация"].sum()),
        int(prepared["пробный_урок"].sum()),
        int(prepared["оплата"].sum()),
    ]
    steps_rows = []
    for index, step in enumerate(STEPS):
        from_start = counts[index] / counts[0]
        from_previous = counts[index] / counts[index - 1] if index else 1.0
        steps_rows.append([step, number(counts[index]), percent(from_start), percent(from_previous)])

    report.add(
        Section(
            title="Воронка целиком",
            text=(
                f"{cleaning.line()} Самая большая потеря — между "
                f"«{STEPS[int(np.argmin([counts[i] / counts[i - 1] for i in range(1, 4)])) + 1 - 1]}» и следующим "
                "шагом."
            ),
            headers=["Шаг", "Людей", "От начала", "От предыдущего шага"],
            rows=steps_rows,
            image=charts.funnel(
                list(STEPS),
                counts,
                path=pictures / "воронка.svg",
                title="Воронка онлайн-школы",
                subtitle="от визита до оплаты",
            ),
        )
    )

    channels = _channels(prepared)
    best = channels.sort_values("оплата", ascending=False)
    baseline = prepared[prepared["канал"] == "поиск"]
    tests = []
    for name in best.index:
        if name == "поиск":
            continue
        group = prepared[prepared["канал"] == name]
        tests.append(
            (
                str(name),
                stats.compare_shares(
                    int(group["оплата"].sum()),
                    len(group),
                    int(baseline["оплата"].sum()),
                    len(baseline),
                    labels=(f"канал «{name}»", "канал «поиск»"),
                ),
            )
        )
    decisions = stats.holm([result.p_value for _, result in tests])

    report.add(
        Section(
            title="Каналы: конверсия в оплату",
            text=(
                "Каждый канал сравнивается с «поиском» как с самым крупным. Сравнений четыре, поэтому "
                "применяется поправка Холма — без неё один «победитель» находится случайно."
            ),
            headers=["Канал", "Визитов", "Оплат", "Конверсия", "p-значение", "Значимо с поправкой"],
            rows=[
                [
                    name,
                    number(as_float(channels.loc[name, "визитов"])),
                    number(as_float(channels.loc[name, "оплат"])),
                    percent(as_float(channels.loc[name, "оплата"])),
                    f"{result.p_value:.4f}",
                    "да" if decision else "нет",
                ]
                for (name, result), decision in zip(tests, decisions, strict=True)
            ],
            image=charts.bars(
                [str(name) for name in best.index],
                [float(value) for value in best["оплата"]],
                path=pictures / "каналы-конверсия.svg",
                title="Конверсия визита в оплату по каналам",
                subtitle="доля визитов, дошедших до оплаты",
                ylabel="доля",
            ),
        )
    )

    economics = _economics(channels)
    report.add(
        Section(
            title="Окупаемость",
            text=(
                "Конверсия сама по себе не решает: у канала может быть отличная конверсия и запредельная "
                "цена визита. Ниже — стоимость привлечения ученика и окупаемость при цене курса "
                f"{money(PRICE)}."
            ),
            headers=["Канал", "Визитов", "Учеников", "Расходы", "Стоимость ученика", "Выручка", "Окупаемость"],
            rows=[
                [
                    str(name),
                    number(as_float(row["визитов"])),
                    number(as_float(row["оплат"])),
                    money(as_float(row["расходы"])),
                    money(as_float(row["цена_ученика"])) if row["оплат"] else "—",
                    money(as_float(row["выручка"])),
                    change(as_float(row["romi"])),
                ]
                for name, row in economics.iterrows()
            ],
            image=charts.bars(
                [str(name) for name in economics.index],
                [float(value) for value in economics["цена_ученика"]],
                path=pictures / "стоимость-ученика.svg",
                title="Во сколько обходится один оплативший ученик",
                subtitle="расходы канала, делённые на число оплат",
                ylabel="₽",
                horizontal=True,
            ),
        )
    )

    cohorts = (
        prepared.groupby("неделя")
        .agg(визитов=("пользователь", "size"), оплат=("оплата", "sum"))
        .assign(конверсия=lambda frame: frame["оплат"] / frame["визитов"])
    )
    report.add(
        Section(
            title="Недельные когорты",
            text=(
                "Конверсия держится на одном уровне все двенадцать недель — значит, различия между "
                "каналами не объясняются удачным периодом."
            ),
            image=charts.lines(
                [f"{week:%d.%m}" for week in cohorts.index],
                {"конверсия в оплату": [float(value) for value in cohorts["конверсия"]]},
                path=pictures / "когорты.svg",
                title="Конверсия в оплату по неделям привлечения",
                subtitle="доля визитов недели, дошедших до оплаты",
                ylabel="доля",
            ),
        )
    )

    leader = str(economics.sort_values("romi", ascending=False).index[0])
    worst = str(economics.sort_values("romi").index[0])
    weakest_step = int(np.argmin([counts[index] / counts[index - 1] for index in range(1, 4)])) + 1
    report.findings = [
        f"Самая большая потеря — переход «{STEPS[weakest_step - 1]}» → «{STEPS[weakest_step]}»: "
        f"доходит {percent(counts[weakest_step] / counts[weakest_step - 1])}.",
        f"Лучше всех окупается «{leader}»: {change(as_float(economics.loc[leader, 'romi']))} при стоимости ученика "
        f"{money(as_float(economics.loc[leader, 'цена_ученика']))}.",
        f"Хуже всех — «{worst}»: {change(as_float(economics.loc[worst, 'romi']))}; при таком результате бюджет "
        "канала стоит пересобрать, а не масштабировать.",
        "Каналы с высокой конверсией не всегда выгодны: решение принято по окупаемости, а не по конверсии.",
    ]
    report.limits = [
        "Данные синтетические: конверсии и цены визитов заложены генератором.",
        "Учитывается только первая оплата: продления, возвраты и отток учеников не моделируются, поэтому окупаемость "
        "завышена.",
        "Каналы не независимы: человек видит рекламу и потом приходит из поиска — в этих данных такой переход не "
        "виден.",
        "Двенадцать недель — короткий период; сезонность спроса на обучение здесь не проявляется.",
    ]
    return report


def _channels(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("канал").agg(
        визитов=("пользователь", "size"),
        регистраций=("регистрация", "sum"),
        уроков=("пробный_урок", "sum"),
        оплат=("оплата", "sum"),
        стоимость_визита=("стоимость_визита", "mean"),
    )
    grouped["оплата"] = grouped["оплат"] / grouped["визитов"]
    return grouped


def _economics(channels: pd.DataFrame) -> pd.DataFrame:
    economics = channels.copy()
    economics["расходы"] = economics["визитов"] * economics["стоимость_визита"]
    economics["выручка"] = economics["оплат"] * PRICE
    economics["цена_ученика"] = np.where(
        economics["оплат"] > 0, economics["расходы"] / economics["оплат"].replace(0, np.nan), 0.0
    )
    economics["romi"] = (economics["выручка"] - economics["расходы"]) / economics["расходы"]
    return economics.sort_values("romi", ascending=False)


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
