"""Исследование 2. От чего зависит цена аренды квартиры.

Разведочный анализ объявлений: что на самом деле объясняет цену — площадь, район или этаж. Здесь
важнее всего аккуратность с выбросами: одно объявление «сдам за 900 000 ₽ в месяц» сдвигает среднее
по району так, что вывод получается обратный правде. Поэтому везде, где можно, используется медиана,
а границы выбросов считаются и показываются явно.
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

NAME: Final = "rent"
TITLE: Final = "От чего зависит цена аренды квартиры"
FOLDER: Final = "02-аренда-квартир"
QUESTION: Final = "Что сильнее влияет на цену аренды: площадь, район или этаж?"

DISTRICTS: Final = {
    "Центральный": 1.45,
    "Приморский": 1.15,
    "Заречный": 1.0,
    "Северный": 0.92,
    "Промышленный": 0.78,
}
ADS: Final = 8_000


def generate(seed: int = data.SEED) -> pd.DataFrame:
    """Объявления об аренде: цена собирается из площади, района и этажа плюс шум."""
    rng = data.generator(seed)
    count = ADS

    names = list(DISTRICTS)
    district = rng.choice(names, size=count, p=[0.14, 0.2, 0.26, 0.24, 0.16])
    coefficient = np.array([DISTRICTS[name] for name in district])

    rooms = rng.choice([1, 2, 3, 4], size=count, p=[0.42, 0.35, 0.18, 0.05])
    area = np.clip(rooms * rng.normal(19, 4, count) + rng.normal(12, 5, count), 18, 200).round(1)
    floors = rng.integers(2, 26, size=count)
    floor = np.array([rng.integers(1, int(top) + 1) for top in floors])
    distance = np.clip(rng.gamma(2.2, 2.1, count), 0.3, 28).round(1)

    base = 480 * area * coefficient
    penalty = np.where((floor == 1) | (floor == floors), 0.93, 1.0)
    price = data.money(base * penalty * (1 - 0.012 * distance) * rng.normal(1, 0.11, count))
    published = np.clip(rng.gamma(2.0, 22, count), 1, 400).round(0)

    frame = pd.DataFrame(
        {
            "район": district,
            "комнат": rooms,
            "площадь": area,
            "этаж": floor,
            "этажей_в_доме": floors,
            "до_центра_км": distance,
            "цена": price,
            "дней_в_публикации": published.astype(int),
        }
    )

    frame = data.spoil_missing(frame, "до_центра_км", 0.05, rng)
    frame = data.spoil_outliers(frame, "цена", 0.003, 12.0, rng)
    return data.spoil_duplicates(frame, 0.012, rng)


def run(out: Path, seed: int = data.SEED) -> Report:
    raw = generate(seed)
    data.save_csv(raw, out / "данные.csv")
    pictures = out / "графики"

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"{len(raw)} объявлений об аренде: район, площадь, число комнат, этаж, расстояние до центра, "
            "цена и срок публикации. Набор воспроизводим по зерну и лежит рядом в «данные.csv»."
        ),
        method=(
            "подготовка данных, границы выбросов по межквартильному размаху, медианы по районам, "
            "корреляция Спирмена и критерий Манна–Уитни для сравнения этажей"
        ),
        prepared_on=datetime.now(UTC).date(),
    )

    checked = quality.inspect(raw)
    headers, rows = checked.table()
    report.add(
        Section(
            title="Что пришло",
            text=checked.summary() + " Расстояние до центра заполняется не всегда — его указывает продавец.",
            headers=headers,
            rows=rows,
        )
    )

    price_bounds = quality.outliers(raw["цена"])
    prepared, cleaning = quality.clean(raw, required=["до_центра_км"], trim=["цена", "площадь"])
    prepared["цена_за_метр"] = data.money((prepared["цена"] / prepared["площадь"]).to_numpy())
    report.add(
        Section(
            title="Что сделали с данными",
            text=(
                f"{cleaning.line()} {price_bounds.line('цена')} — в объявлениях это опечатки в нулях. "
                "Дальше везде считается медиана: одно объявление за миллион смещает среднее, медиану — нет."
            ),
        )
    )

    by_district = (
        prepared.groupby("район")
        .agg(объявлений=("цена", "size"), медиана=("цена", "median"), за_метр=("цена_за_метр", "median"))
        .sort_values("за_метр", ascending=False)
    )
    report.add(
        Section(
            title="Районы",
            text=(
                "Разница между самым дорогим и самым дешёвым районом — "
                f"{percent(float(by_district['за_метр'].iloc[0] / by_district['за_метр'].iloc[-1]) - 1)} "
                "по цене за метр."
            ),
            headers=["Район", "Объявлений", "Медиана цены", "Медиана за м²"],
            rows=[
                [
                    str(name),
                    number(float(row["объявлений"])),
                    money(float(row["медиана"])),
                    money(float(row["за_метр"])),
                ]
                for name, row in by_district.iterrows()
            ],
            image=charts.bars(
                [str(name) for name in by_district.index],
                [float(value) for value in by_district["за_метр"]],
                path=pictures / "цена-за-метр-по-районам.svg",
                title="Медианная цена аренды за квадратный метр",
                subtitle="по районам, ₽ в месяц",
                ylabel="₽ за м²",
            ),
        )
    )

    correlation = float(scipy_stats.spearmanr(prepared["площадь"], prepared["цена"]).statistic)
    distance_correlation = float(scipy_stats.spearmanr(prepared["до_центра_км"], prepared["цена_за_метр"]).statistic)
    report.add(
        Section(
            title="Площадь и расстояние",
            text=(
                f"Связь площади и цены — {correlation:.2f} по Спирмену: это сильная монотонная связь. "
                f"Связь расстояния до центра и цены за метр — {distance_correlation:.2f}: слабее, "
                "но заметна и направлена вниз."
            ),
            image=charts.scatter(
                [float(value) for value in prepared["площадь"]],
                [float(value) for value in prepared["цена"]],
                path=pictures / "площадь-и-цена.svg",
                title="Площадь и цена аренды",
                subtitle="каждая точка — объявление",
                xlabel="площадь, м²",
                ylabel="цена, ₽ в месяц",
            ),
        )
    )

    edge = prepared[(prepared["этаж"] == 1) | (prepared["этаж"] == prepared["этажей_в_доме"])]["цена_за_метр"]
    middle = prepared[(prepared["этаж"] != 1) & (prepared["этаж"] != prepared["этажей_в_доме"])]["цена_за_метр"]
    floor_test = stats.compare_distributions(
        [float(value) for value in edge],
        [float(value) for value in middle],
        labels=("первый и последний этаж", "средние этажи"),
    )
    report.add(
        Section(
            title="Этаж",
            text=floor_test.conclusion,
            image=charts.boxes(
                {
                    "первый и последний": [float(value) for value in edge],
                    "средние этажи": [float(value) for value in middle],
                },
                path=pictures / "этажи.svg",
                title="Цена за метр по этажам",
                subtitle="медиана, квартили и усы",
                ylabel="₽ за м²",
            ),
        )
    )

    speed = prepared.groupby("район")["дней_в_публикации"].median().sort_values()
    report.add(
        Section(
            title="Сколько висит объявление",
            text=(
                f"Быстрее всего уходят квартиры в районе «{speed.index[0]}» — медиана "
                f"{number(float(speed.iloc[0]))} дней; дольше всего в «{speed.index[-1]}» — "
                f"{number(float(speed.iloc[-1]))} дней."
            ),
            image=charts.histogram(
                [float(value) for value in prepared["дней_в_публикации"]],
                path=pictures / "срок-публикации.svg",
                title="Сколько дней объявление висит до сдачи",
                subtitle="распределение по всем районам",
                bins=40,
                xlabel="дней",
            ),
        )
    )

    report.findings = [
        f"Площадь объясняет цену лучше всего: связь по Спирмену {correlation:.2f}.",
        "Район добавляет к цене метра до "
        f"{percent(float(by_district['за_метр'].iloc[0] / by_district['за_метр'].iloc[-1]) - 1)}.",
        floor_test.conclusion,
        f"Половина объявлений уходит быстрее {number(float(prepared['дней_в_публикации'].median()))} дней.",
    ]
    report.limits = [
        "Данные синтетические: зависимости заложены генератором, поэтому числа иллюстрируют метод, а не рынок города.",
        "В наборе нет ремонта, мебели и вида из окна — а они влияют на цену не меньше этажа.",
        "Цена в объявлении не равна цене сделки: торг в данных не отражён.",
        "Выбросы удалены по межквартильному размаху; для дорогого сегмента это спорный приём, "
        "и при другой задаче их пришлось бы анализировать отдельно.",
    ]
    return report


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
