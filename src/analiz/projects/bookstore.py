"""Исследование 8. Продажи книжного магазина: то же самое, но на SQL.

Отдельная работа именно на SQL нужна потому, что на собеседовании аналитика спрашивают не pandas,
а оконные функции и подзапросы. Здесь пять запросов, которые закрывают почти весь типовой список:
накопленная доля (ABC-анализ), ранг внутри группы, сравнение с предыдущим месяцем через `lag`,
когорты по месяцу первой покупки и доля повторных покупателей.

Запросы приведены в отчёте целиком: это и есть предмет разговора на собеседовании.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import duckdb
import numpy as np
import pandas as pd

from analiz import charts, data
from analiz.projects import Project
from analiz.report import Report, Section, money, number, percent

NAME: Final = "bookstore"
TITLE: Final = "Продажи книжного магазина: разбор на SQL"
FOLDER: Final = "08-sql-книжный-магазин"
QUESTION: Final = "Какие книги дают выручку, как растут продажи и возвращаются ли покупатели?"

GENRES: Final = ("фантастика", "детективы", "нон-фикшн", "детские", "поэзия", "бизнес")
AUTHORS: Final = (
    "Стругацкие",
    "Пелевин",
    "Акунин",
    "Улицкая",
    "Иванов",
    "Рубина",
    "Сорокин",
    "Быков",
    "Водолазкин",
    "Яхина",
)
BOOKS: Final = 220
CUSTOMERS: Final = 1_800
ORDERS: Final = 9_000
START: Final = date(2025, 1, 1)
DAYS: Final = 540


def generate(seed: int = data.SEED) -> dict[str, pd.DataFrame]:
    """Три таблицы: книги, заказы и позиции заказов — как в обычной базе магазина."""
    rng = data.generator(seed)

    books = pd.DataFrame(
        {
            "книга_id": np.arange(1, BOOKS + 1),
            "название": [f"Книга №{number}" for number in range(1, BOOKS + 1)],
            "автор": rng.choice(AUTHORS, size=BOOKS),
            "жанр": rng.choice(GENRES, size=BOOKS, p=[0.24, 0.2, 0.18, 0.16, 0.08, 0.14]),
            "цена": data.money(np.clip(rng.normal(780, 320, BOOKS), 180, 3_500)),
        }
    )

    # Популярность книг распределена неравномерно: несколько бестселлеров тянут выручку.
    popularity = rng.pareto(2.6, BOOKS) + 0.6
    weights = popularity / popularity.sum()

    customer_first = rng.integers(0, DAYS - 30, size=CUSTOMERS)
    order_customer = rng.choice(np.arange(1, CUSTOMERS + 1), size=ORDERS, p=_repeat_weights(rng, CUSTOMERS))
    offsets = np.array([int(customer_first[client - 1]) for client in order_customer])
    order_day = np.clip(offsets + rng.integers(0, 120, size=ORDERS), 0, DAYS - 1)

    orders = pd.DataFrame(
        {
            "заказ_id": np.arange(1, ORDERS + 1),
            "клиент_id": order_customer,
            "дата": [START + pd.Timedelta(days=int(value)) for value in order_day],
            "канал": rng.choice(["сайт", "приложение", "магазин"], size=ORDERS, p=[0.5, 0.3, 0.2]),
        }
    )

    lines: list[dict[str, object]] = []
    for order_id in orders["заказ_id"]:
        for _ in range(int(rng.integers(1, 4))):
            book = int(rng.choice(books["книга_id"], p=weights))
            quantity = int(rng.integers(1, 3))
            price = float(books.loc[books["книга_id"] == book, "цена"].iloc[0])
            discount = float(rng.choice([0.0, 0.0, 0.05, 0.1, 0.15]))
            lines.append(
                {
                    "заказ_id": int(order_id),
                    "книга_id": book,
                    "количество": quantity,
                    "цена": round(price * (1 - discount), 2),
                }
            )

    return {"книги": books, "заказы": orders, "позиции": pd.DataFrame(lines)}


def _repeat_weights(rng: np.random.Generator, customers: int) -> np.ndarray:
    """Часть клиентов покупает много раз, большинство — один-два: так и бывает в рознице."""
    weights = rng.pareto(2.4, customers) + 0.5
    return weights / weights.sum()


QUERY_TOP: Final = """
WITH выручка AS (
    SELECT п.книга_id,
           к.название,
           к.автор,
           к.жанр,
           sum(п.количество * п.цена) AS выручка
    FROM позиции п
    JOIN книги к USING (книга_id)
    GROUP BY 1, 2, 3, 4
)
SELECT название,
       автор,
       жанр,
       выручка,
       выручка / sum(выручка) OVER () AS доля,
       sum(выручка) OVER (ORDER BY выручка DESC) / sum(выручка) OVER () AS накопленная_доля
FROM выручка
ORDER BY выручка DESC
LIMIT 10
"""

QUERY_ABC: Final = """
WITH выручка AS (
    SELECT п.книга_id, sum(п.количество * п.цена) AS выручка
    FROM позиции п GROUP BY 1
),
накопление AS (
    SELECT книга_id,
           выручка,
           sum(выручка) OVER (ORDER BY выручка DESC) / sum(выручка) OVER () AS накопленная_доля
    FROM выручка
)
SELECT CASE WHEN накопленная_доля <= 0.8 THEN 'A'
            WHEN накопленная_доля <= 0.95 THEN 'B'
            ELSE 'C' END AS группа,
       count(*) AS книг,
       sum(выручка) AS выручка
FROM накопление
GROUP BY 1
ORDER BY 1
"""

QUERY_AUTHORS: Final = """
SELECT жанр,
       автор,
       выручка,
       ранг
FROM (
    SELECT к.жанр,
           к.автор,
           sum(п.количество * п.цена) AS выручка,
           rank() OVER (PARTITION BY к.жанр ORDER BY sum(п.количество * п.цена) DESC) AS ранг
    FROM позиции п
    JOIN книги к USING (книга_id)
    GROUP BY 1, 2
)
WHERE ранг <= 2
ORDER BY жанр, ранг
"""

QUERY_MONTHS: Final = """
WITH помесячно AS (
    SELECT date_trunc('month', з.дата) AS месяц,
           sum(п.количество * п.цена) AS выручка,
           count(DISTINCT з.заказ_id) AS заказов
    FROM позиции п
    JOIN заказы з USING (заказ_id)
    GROUP BY 1
)
SELECT месяц,
       выручка,
       заказов,
       lag(выручка) OVER (ORDER BY месяц) AS прошлый_месяц,
       выручка / nullif(lag(выручка) OVER (ORDER BY месяц), 0) - 1 AS рост
FROM помесячно
ORDER BY месяц
"""

QUERY_COHORTS: Final = """
WITH первая_покупка AS (
    SELECT клиент_id, min(date_trunc('month', дата)) AS когорта
    FROM заказы GROUP BY 1
),
заказы_клиентов AS (
    SELECT з.клиент_id,
           п.когорта,
           date_diff('month', п.когорта, date_trunc('month', з.дата)) AS месяц_жизни
    FROM заказы з JOIN первая_покупка п USING (клиент_id)
)
SELECT когорта,
       count(DISTINCT клиент_id) FILTER (WHERE месяц_жизни = 0) AS клиентов,
       count(DISTINCT клиент_id) FILTER (WHERE месяц_жизни = 1) AS вернулись_через_месяц,
       count(DISTINCT клиент_id) FILTER (WHERE месяц_жизни = 2) AS вернулись_через_два
FROM заказы_клиентов
GROUP BY 1
ORDER BY 1
"""


def run(out: Path, seed: int = data.SEED) -> Report:
    tables = generate(seed)
    for name, frame in tables.items():
        data.save_csv(frame, out / f"{name}.csv")
    pictures = out / "графики"

    connection = duckdb.connect(":memory:")
    for name, frame in tables.items():
        connection.register(name, frame)

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"три таблицы: {len(tables['книги'])} книг, {len(tables['заказы'])} заказов и "
            f"{len(tables['позиции'])} позиций за полтора года. Наборы воспроизводимы по зерну и лежат рядом в CSV."
        ),
        method="пять SQL-запросов: оконные функции, накопленные суммы, ранги внутри групп, сравнение с предыдущим "
        "месяцем и когорты",
        prepared_on=datetime.now(UTC).date(),
    )

    top = connection.execute(QUERY_TOP).fetchdf()
    report.add(
        Section(
            title="Топ книг по выручке",
            text=(
                "Накопленная доля считается оконной функцией `sum(...) OVER (ORDER BY выручка DESC)`: "
                "это тот самый приём, который просят написать на собеседовании.\n\n```sql" + QUERY_TOP + "```"
            ),
            headers=["Книга", "Автор", "Жанр", "Выручка", "Доля", "Накопленная доля"],
            rows=[
                [
                    str(row["название"]),
                    str(row["автор"]),
                    str(row["жанр"]),
                    money(float(row["выручка"])),
                    percent(float(row["доля"])),
                    percent(float(row["накопленная_доля"])),
                ]
                for _, row in top.iterrows()
            ],
            image=charts.bars(
                [str(value) for value in top["название"]],
                [float(value) for value in top["выручка"]],
                path=pictures / "топ-книг.svg",
                title="Десять книг с наибольшей выручкой",
                subtitle="за весь период",
                ylabel="₽",
                horizontal=True,
            ),
        )
    )

    abc = connection.execute(QUERY_ABC).fetchdf()
    total = float(abc["выручка"].sum())
    report.add(
        Section(
            title="ABC-анализ ассортимента",
            text=(
                "Группа A — книги, которые дают первые 80% выручки. Это стандартный приём управления "
                "ассортиментом: закупки и полки распределяются по этим группам.\n\n```sql" + QUERY_ABC + "```"
            ),
            headers=["Группа", "Книг", "Выручка", "Доля выручки"],
            rows=[
                [
                    str(row["группа"]),
                    number(float(row["книг"])),
                    money(float(row["выручка"])),
                    percent(float(row["выручка"]) / total),
                ]
                for _, row in abc.iterrows()
            ],
            image=charts.bars(
                [str(value) for value in abc["группа"]],
                [float(value) for value in abc["книг"]],
                path=pictures / "abc.svg",
                title="Сколько книг в каждой группе ABC",
                subtitle="A — первые 80% выручки, B — до 95%, C — остальное",
                ylabel="книг",
            ),
        )
    )

    authors = connection.execute(QUERY_AUTHORS).fetchdf()
    report.add(
        Section(
            title="Лидеры по жанрам",
            text="Ранг внутри группы — `rank() OVER (PARTITION BY жанр ORDER BY выручка DESC)`.\n\n```sql"
            + QUERY_AUTHORS
            + "```",
            headers=["Жанр", "Автор", "Выручка", "Место в жанре"],
            rows=[
                [str(row["жанр"]), str(row["автор"]), money(float(row["выручка"])), str(int(row["ранг"]))]
                for _, row in authors.iterrows()
            ],
        )
    )

    months = connection.execute(QUERY_MONTHS).fetchdf()
    report.add(
        Section(
            title="Помесячная динамика",
            text="Сравнение с предыдущим месяцем делается через `lag` — без него пришлось бы соединять таблицу саму с "
            "собой.\n\n```sql" + QUERY_MONTHS + "```",
            image=charts.lines(
                [f"{value:%m.%Y}" for value in months["месяц"]],
                {"выручка": [float(value) for value in months["выручка"]]},
                path=pictures / "помесячно.svg",
                title="Выручка по месяцам",
                subtitle="все каналы вместе",
                ylabel="₽",
            ),
        )
    )

    cohorts = connection.execute(QUERY_COHORTS).fetchdf()
    returned = float(cohorts["вернулись_через_месяц"].sum()) / max(float(cohorts["клиентов"].sum()), 1)
    report.add(
        Section(
            title="Когорты покупателей",
            text=(
                f"Через месяц после первой покупки возвращается {percent(returned)} клиентов. "
                "Когорта считается как месяц первой покупки, а срок жизни — через `date_diff`.\n\n```sql"
                + QUERY_COHORTS
                + "```"
            ),
            headers=["Когорта", "Клиентов", "Вернулись через месяц", "Через два месяца"],
            rows=[
                [
                    f"{row['когорта']:%m.%Y}",
                    number(float(row["клиентов"])),
                    number(float(row["вернулись_через_месяц"])),
                    number(float(row["вернулись_через_два"])),
                ]
                for _, row in cohorts.iterrows()
            ],
        )
    )

    connection.close()

    group_a = abc[abc["группа"] == "A"]
    report.findings = [
        f"Первые 80% выручки дают {number(float(group_a['книг'].iloc[0]))} книг из {len(tables['книги'])} "
        f"({percent(float(group_a['книг'].iloc[0]) / len(tables['книги']))} ассортимента) — это группа A, "
        "с которой работают в первую очередь.",
        f"Самая доходная книга приносит {percent(float(top['доля'].iloc[0]))} всей выручки.",
        f"Через месяц после первой покупки возвращается {percent(returned)} клиентов.",
        "Все расчёты сделаны одним SQL: оконные функции закрывают и накопленные доли, и ранги, и сравнение с прошлым "
        "месяцем.",
    ]
    report.limits = [
        "Данные синтетические: неравномерность продаж заложена генератором (распределение Парето).",
        "Возвраты, отмены и себестоимость не моделируются, поэтому «выручка» здесь — валовая.",
        "Когорты считаются по календарным месяцам; клиент, купивший 31-го и 1-го, попадёт в разные месяцы жизни.",
        "Запросы написаны под DuckDB: в PostgreSQL `date_diff` называется иначе, а `FILTER` поддерживается не везде.",
    ]
    return report


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
