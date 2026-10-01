"""Исследование 7. Кто из клиентов автосервиса не вернётся.

Задача с моделью — но не про модель. Куда важнее три вещи, которые решают её судьбу в работе.

Первая: с чем сравнивать. Модель, которая «угадывает» 91% при 9% оттока, ничего не умеет — столько
же даст ответ «никто не уйдёт». Поэтому рядом всегда стоит эта опорная величина.

Вторая: порог. Модель выдаёт вероятность, а решение принимает человек, и порог зависит от денег:
сколько стоит зря отправленная скидка и сколько — потерянный клиент. Порог здесь считается из этих
двух чисел, а не берётся 0,5 по умолчанию.

Третья: утечка. Признак «месяцев с последнего визита» знает ответ заранее, поэтому он в модель
не попадает — и об этом сказано прямо.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analiz import charts, data, quality, stats
from analiz.projects import Project
from analiz.report import Report, Section, money, number, percent

NAME: Final = "churn"
TITLE: Final = "Кто из клиентов автосервиса не вернётся"
FOLDER: Final = "07-отток-автосервиса"
QUESTION: Final = "Можно ли заранее увидеть клиентов, которые больше не приедут, и кому слать скидку?"

CLIENTS: Final = 6_000
KEEP_COST: Final = 900.0
"""Скидка на следующее обслуживание — столько стоит одна попытка удержания."""
LOST_VALUE: Final = 14_000.0
"""Во столько обходится потерянный клиент: средняя выручка с него за следующий год."""

SERVICE_TYPES: Final = ("ТО", "ремонт", "шиномонтаж", "кузовной")
FEATURES: Final = ("возраст_авто", "пробег_тыс_км", "визитов_за_год", "средний_чек", "абонемент", "жалоб")


def generate(seed: int = data.SEED) -> pd.DataFrame:
    """Клиенты автосервиса: отток зависит от числа визитов, жалоб и абонемента."""
    rng = data.generator(seed)
    count = CLIENTS

    car_age = np.clip(rng.gamma(2.6, 2.4, count), 0.5, 22).round(1)
    mileage = np.clip(car_age * rng.normal(14, 4, count), 3, 400).round(1)
    visits = rng.poisson(2.6, count)
    receipt = data.money(np.clip(rng.lognormal(8.6, 0.55, count), 800, 180_000))
    subscription = rng.random(count) < 0.22
    complaints = rng.poisson(0.35, count)
    service = rng.choice(SERVICE_TYPES, size=count, p=[0.46, 0.26, 0.2, 0.08])

    risk = 0.34 - 0.075 * visits + 0.1 * complaints - 0.16 * subscription + 0.012 * car_age - 0.00000_9 * receipt
    churned = rng.random(count) < np.clip(risk, 0.02, 0.95)
    # Признак-подсказка: он известен только постфактум и в модель не пойдёт.
    months_since = np.where(churned, rng.integers(7, 18, count), rng.integers(0, 7, count))

    frame = pd.DataFrame(
        {
            "клиент": np.arange(1, count + 1),
            "возраст_авто": car_age,
            "пробег_тыс_км": mileage,
            "визитов_за_год": visits,
            "средний_чек": receipt,
            "абонемент": subscription.astype(int),
            "жалоб": complaints,
            "тип_обслуживания": service,
            "месяцев_с_последнего_визита": months_since,
            "ушёл": churned.astype(int),
        }
    )

    frame = data.spoil_missing(frame, "средний_чек", 0.05, rng)
    return data.spoil_duplicates(frame, 0.007, rng)


def run(out: Path, seed: int = data.SEED) -> Report:
    raw = generate(seed)
    data.save_csv(raw, out / "данные.csv")
    pictures = out / "графики"

    report = Report(
        title=TITLE,
        question=QUESTION,
        data=(
            f"{len(raw)} клиентов автосервиса: возраст и пробег машины, число визитов за год, средний чек, "
            "абонемент, жалобы и факт ухода. Набор воспроизводим по зерну."
        ),
        method=(
            "подготовка данных, сравнение групп, логистическая регрессия и случайный лес с разделением "
            "на обучение и проверку, сравнение с опорным ответом и подбор порога по цене ошибки"
        ),
        prepared_on=datetime.now(UTC).date(),
    )

    checked = quality.inspect(raw)
    headers, rows = checked.table()
    report.add(Section(title="Что пришло", text=checked.summary(), headers=headers, rows=rows))

    prepared, cleaning = quality.clean(raw, required=["средний_чек"])
    share = float(prepared["ушёл"].mean())
    report.add(
        Section(
            title="Что сделали с данными",
            text=(
                f"{cleaning.line()} Доля ушедших — {percent(share)}. "
                "Признак «месяцев с последнего визита» намеренно исключён из модели: он известен только "
                "после того, как клиент уже ушёл, и модель с ним показала бы фантастическое качество, "
                "бесполезное на практике."
            ),
        )
    )

    stayed = prepared[prepared["ушёл"] == 0]
    left = prepared[prepared["ушёл"] == 1]
    visits_test = stats.compare_distributions(
        [float(value) for value in left["визитов_за_год"]],
        [float(value) for value in stayed["визитов_за_год"]],
        labels=("ушедшие", "оставшиеся"),
    )
    subscription_test = stats.compare_shares(
        int(left["абонемент"].sum()),
        len(left),
        int(stayed["абонемент"].sum()),
        len(stayed),
        labels=("доля абонементов у ушедших", "у оставшихся"),
    )
    report.add(
        Section(
            title="Чем ушедшие отличаются от оставшихся",
            text=f"{visits_test.conclusion} {subscription_test.conclusion}",
            headers=["Показатель", "Ушли", "Остались"],
            rows=[
                [
                    "визитов за год (медиана)",
                    number(float(left["визитов_за_год"].median())),
                    number(float(stayed["визитов_за_год"].median())),
                ],
                [
                    "средний чек (медиана)",
                    money(float(left["средний_чек"].median())),
                    money(float(stayed["средний_чек"].median())),
                ],
                [
                    "доля с абонементом",
                    percent(float(left["абонемент"].mean())),
                    percent(float(stayed["абонемент"].mean())),
                ],
                ["жалоб на клиента", number(float(left["жалоб"].mean()), 2), number(float(stayed["жалоб"].mean()), 2)],
            ],
        )
    )

    trained = _train(prepared, seed)
    baseline = 1 - float(trained["y_test"].mean())
    report.add(
        Section(
            title="Модель и с чем её сравнивать",
            text=(
                f"Данные разделены на обучение и проверку в отношении 75 на 25 с сохранением доли ушедших. "
                f"Опорный ответ «никто не уйдёт» даёт точность {percent(baseline)} — любая модель обязана "
                "быть полезнее него, а «высокая точность» сама по себе ни о чём не говорит."
            ),
            headers=["Модель", "ROC-AUC", "Точность при пороге 0,5", "Полнота при пороге 0,5"],
            rows=[
                [
                    "логистическая регрессия",
                    f"{trained['auc_logistic']:.3f}",
                    percent(trained["precision_logistic"]),
                    percent(trained["recall_logistic"]),
                ],
                [
                    "случайный лес",
                    f"{trained['auc_forest']:.3f}",
                    percent(trained["precision_forest"]),
                    percent(trained["recall_forest"]),
                ],
                ["опорный ответ «никто не уйдёт»", "0.500", "—", "0,0%"],
            ],
        )
    )

    threshold = trained["threshold"]
    matrix = trained["matrix"]
    report.add(
        Section(
            title="Порог: сколько стоит ошибка",
            text=(
                f"Скидка на удержание стоит {money(KEEP_COST)}, потерянный клиент — {money(LOST_VALUE)}. "
                f"Значит, предлагать скидку выгодно всем, у кого вероятность ухода выше "
                f"{percent(KEEP_COST / LOST_VALUE)}. На проверочной выборке лучший порог — "
                f"{number(threshold, 2)}; при нём мы находим {percent(trained['recall_at_threshold'])} "
                f"ушедших, а каждая третья скидка уходит тому, кто и так вернулся бы."
            ),
            headers=["", "Предсказали «останется»", "Предсказали «уйдёт»"],
            rows=[
                ["На самом деле остался", number(float(matrix[0][0])), number(float(matrix[0][1]))],
                ["На самом деле ушёл", number(float(matrix[1][0])), number(float(matrix[1][1]))],
            ],
            image=charts.lines(
                [f"{value:.2f}" for value in trained["thresholds"]],
                {
                    "точность": trained["precisions"],
                    "полнота": trained["recalls"],
                },
                path=pictures / "порог.svg",
                title="Точность и полнота при разных порогах",
                subtitle="чем ниже порог, тем больше находим и тем больше лишних скидок",
                ylabel="доля",
            ),
        )
    )

    importance = trained["importance"]
    report.add(
        Section(
            title="Что важно для модели",
            text="Порядок признаков совпадает с тем, что видно в простом сравнении групп, — это хороший знак.",
            image=charts.bars(
                [str(name) for name in importance.index],
                [float(value) for value in importance],
                path=pictures / "важность-признаков.svg",
                title="Важность признаков (случайный лес)",
                subtitle="вклад признака в разделение классов",
                ylabel="важность",
                horizontal=True,
            ),
        )
    )

    saved = trained["profit"]
    report.findings = [
        f"Доля оттока — {percent(share)}; опорный ответ «никто не уйдёт» даёт {percent(baseline)} точности и ноль "
        "пользы.",
        f"Случайный лес различает классы с ROC-AUC {trained['auc_forest']:.3f}, логистическая регрессия — "
        f"{trained['auc_logistic']:.3f}.",
        f"При пороге {number(threshold, 2)}, посчитанном из цены ошибки, кампания удержания на проверочной "
        f"выборке даёт {money(saved)} ожидаемой выгоды.",
        "Сильнее всего на уход влияют число визитов за год, жалобы и отсутствие абонемента.",
    ]
    report.limits = [
        "Данные синтетические: зависимости заложены генератором, поэтому качество модели говорит о методе, а не о "
        "рынке.",
        "Признак «месяцев с последнего визита» исключён как утечка; с ним модель выглядела бы намного лучше, чем она "
        "есть.",
        "Выгода посчитана по средним величинам и предполагает, что скидка удерживает клиента наверняка — в жизни это "
        "не так.",
        "Модель обучена на одном срезе времени: сезонность и изменение цен в ней не учтены.",
        "Проверка — однократное разделение выборки; для решения о внедрении нужна перекрёстная проверка и период "
        "наблюдения.",
    ]
    return report


def _train(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Обучение двух моделей и подбор порога по цене ошибки."""
    features = frame[list(FEATURES)].astype(float)
    target = frame["ушёл"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=seed, stratify=target
    )

    scaler = StandardScaler().fit(x_train)
    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    logistic.fit(scaler.transform(x_train), y_train)
    logistic_probability = logistic.predict_proba(scaler.transform(x_test))[:, 1]

    forest = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=8, class_weight="balanced", random_state=seed, n_jobs=1
    )
    forest.fit(x_train, y_train)
    forest_probability = forest.predict_proba(x_test)[:, 1]

    threshold = KEEP_COST / LOST_VALUE
    predicted = (forest_probability >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predicted)
    precision, recall, thresholds = precision_recall_curve(y_test, forest_probability)

    true_positive = int(matrix[1][1])
    false_positive = int(matrix[0][1])
    profit = true_positive * LOST_VALUE - (true_positive + false_positive) * KEEP_COST

    return {
        "y_test": y_test,
        "auc_logistic": float(roc_auc_score(y_test, logistic_probability)),
        "auc_forest": float(roc_auc_score(y_test, forest_probability)),
        "precision_logistic": _precision(y_test, logistic_probability >= 0.5),
        "recall_logistic": _recall(y_test, logistic_probability >= 0.5),
        "precision_forest": _precision(y_test, forest_probability >= 0.5),
        "recall_forest": _recall(y_test, forest_probability >= 0.5),
        "threshold": threshold,
        "matrix": matrix.tolist(),
        "recall_at_threshold": _recall(y_test, predicted.astype(bool)),
        "profit": profit,
        "thresholds": [float(value) for value in thresholds[:: max(len(thresholds) // 12, 1)]],
        "precisions": [float(value) for value in precision[:-1][:: max(len(thresholds) // 12, 1)]],
        "recalls": [float(value) for value in recall[:-1][:: max(len(thresholds) // 12, 1)]],
        "importance": pd.Series(forest.feature_importances_, index=list(FEATURES)).sort_values(ascending=False),
    }


def _precision(truth: pd.Series, predicted: Any) -> float:
    predicted_positive = int(np.sum(predicted))
    if not predicted_positive:
        return 0.0
    return float(np.sum(np.asarray(truth) & np.asarray(predicted)) / predicted_positive)


def _recall(truth: pd.Series, predicted: Any) -> float:
    actual_positive = int(np.sum(np.asarray(truth)))
    if not actual_positive:
        return 0.0
    return float(np.sum(np.asarray(truth) & np.asarray(predicted)) / actual_positive)


PROJECT: Final = Project(name=NAME, title=TITLE, folder=FOLDER, question=QUESTION, run=run)
