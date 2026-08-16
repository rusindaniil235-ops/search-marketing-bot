"""Explainable keyword-based assistant for search-marketing analytics.

The project intentionally uses only the Python standard library. It does not
connect to corporate systems and does not contain internal data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class Intent:
    name: str
    title: str
    patterns: tuple[str, ...]
    keywords: dict[str, float]
    answer: str


@dataclass(frozen=True)
class ChatResponse:
    intent: str
    title: str
    confidence: float
    answer: str
    matched_keywords: tuple[str, ...]
    alternatives: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FALLBACK_ANSWER = (
    "Я не смогла уверенно определить тему. Спросите про поисковый охват, "
    "конверсию, органический и платный трафик, атрибуцию, аномалии, "
    "качество данных, эксперименты или аналитический отчёт."
)


INTENTS: tuple[Intent, ...] = (
    Intent(
        name="greeting",
        title="Приветствие",
        patterns=("привет", "добрый день", "доброе утро", "здравствуйте"),
        keywords={
            "привет": 2.0,
            "здравств*": 2.0,
            "добрый": 1.0,
            "утро": 0.8,
            "начнем": 1.3,
        },
        answer=(
            "Здравствуйте! Я учебный помощник аналитика маркетинга Поиска. "
            "Могу объяснить базовые метрики и предложить порядок проверки данных."
        ),
    ),
    Intent(
        name="team_scope",
        title="Задачи аналитики маркетинга Поиска",
        patterns=(
            "чем занимается команда",
            "задачи аналитика маркетинга поиска",
            "что делает аналитик",
        ),
        keywords={
            "команд*": 1.1,
            "аналитик*": 1.2,
            "маркетинг*": 1.0,
            "задач*": 1.0,
            "занимает*": 1.1,
            "работ*": 0.5,
            "направлен*": 1.0,
            "устроен*": 1.0,
            "отвеча*": 1.0,
        },
        answer=(
            "В учебной функциональной модели аналитик маркетинга Поиска оценивает "
            "привлечение аудитории, поисковый охват, конверсию, вклад каналов, "
            "качество данных и результаты экспериментов. Модель описывает функции, "
            "а не официальную внутреннюю оргструктуру Яндекса."
        ),
    ),
    Intent(
        name="search_reach",
        title="Поисковый охват",
        patterns=(
            "что такое поисковый охват",
            "почему стало меньше показов",
            "как оценить видимость в поиске",
        ),
        keywords={
            "охват*": 2.2,
            "показ*": 1.5,
            "impression*": 1.8,
            "видимост*": 1.7,
            "аудитор*": 0.9,
            "reach": 1.8,
            "выдач*": 1.7,
            "увид*": 1.2,
            "реже": 1.0,
        },
        answer=(
            "Поисковый охват показывает, сколько раз продукт или предложение было "
            "доступно аудитории в поисковом контексте. При снижении сначала разделяют "
            "изменение спроса, видимости и состава аудитории, а уже затем проверяют "
            "конверсию — эти механизмы нельзя смешивать."
        ),
    ),
    Intent(
        name="conversion",
        title="Конверсия",
        patterns=(
            "как посчитать конверсию",
            "что означает cvr",
            "переход из показа в целевое действие",
        ),
        keywords={
            "конверс*": 2.3,
            "cvr": 2.2,
            "целев*": 1.0,
            "действ*": 0.7,
            "переход*": 0.9,
            "скачиван*": 0.8,
            "установк*": 0.8,
            "воронк*": 1.2,
            "доля перехода": 2.0,
        },
        answer=(
            "Конверсия — отношение числа целевых действий к числу возможностей их "
            "совершить. Числитель и знаменатель должны относиться к одному периоду, "
            "сегменту и единице анализа. Перед сравнением проверяют дедупликацию, "
            "опоздание событий и одинаковую методологию расчёта."
        ),
    ),
    Intent(
        name="traffic_sources",
        title="Каналы привлечения",
        patterns=(
            "органический и платный трафик",
            "чем отличается paid от organic",
            "какие бывают каналы привлечения",
        ),
        keywords={
            "органич*": 2.0,
            "платн*": 1.8,
            "paid": 1.9,
            "organic": 1.9,
            "канал*": 1.3,
            "источник*": 1.0,
            "реклам*": 1.1,
            "трафик*": 1.2,
            "привлечен*": 1.0,
        },
        answer=(
            "Органический трафик получают без прямой оплаты за конкретный переход, "
            "платный — из рекламных размещений. В анализе важно использовать одно "
            "правило атрибуции и отдельно показывать смешанные или неопределённые "
            "источники, иначе вклад каналов будет искажён."
        ),
    ),
    Intent(
        name="attribution",
        title="Атрибуция",
        patterns=(
            "что такое атрибуция",
            "как определить источник привлечения",
            "последний клик или первый клик",
        ),
        keywords={
            "атрибут*": 2.3,
            "присво*": 1.1,
            "источник*": 0.8,
            "последн*": 0.8,
            "перв*": 0.5,
            "клик*": 0.9,
            "канал*": 0.6,
            "засчит*": 1.8,
            "модел*": 0.8,
        },
        answer=(
            "Атрибуция — правило, по которому целевое действие связывают с каналом "
            "или контактом. До сравнения каналов фиксируют модель атрибуции, окно, "
            "идентификатор пользователя и порядок событий. Разные модели дают разные, "
            "но не обязательно ошибочные результаты."
        ),
    ),
    Intent(
        name="anomaly",
        title="Разбор аномалии",
        patterns=(
            "почему упала метрика",
            "как исследовать аномалию",
            "резкий рост показателя",
        ),
        keywords={
            "аномал*": 2.2,
            "паден*": 1.7,
            "упал*": 1.6,
            "просел*": 1.6,
            "снижен*": 1.5,
            "скачок*": 1.5,
            "вырос*": 1.9,
            "рост*": 1.0,
            "измен*": 0.6,
            "метрик*": 0.6,
            "необыч*": 1.5,
        },
        answer=(
            "Разбор аномалии начинают с проверки определения метрики и свежести "
            "данных. Затем раскладывают изменение по этапам воронки, сегментам и "
            "времени, проверяют технические изменения и только после этого обсуждают "
            "причины. Наблюдаемый механизм нужно отделять от недоказанной первопричины."
        ),
    ),
    Intent(
        name="data_quality",
        title="Качество данных",
        patterns=(
            "как проверить качество данных",
            "в таблице появились дубликаты",
            "много пропусков в данных",
        ),
        keywords={
            "качеств*": 1.8,
            "данн*": 0.7,
            "дублик*": 2.0,
            "пропуск*": 1.8,
            "null": 1.8,
            "схем*": 1.2,
            "валид*": 1.5,
            "сверк*": 1.4,
            "полнот*": 1.2,
            "уникальн*": 1.1,
            "не сход*": 1.7,
            "потер*": 1.2,
            "строк*": 0.7,
            "объединен*": 0.8,
        },
        answer=(
            "Минимальная проверка качества включает схему и типы, полноту, NULL, "
            "уникальность ключа, дубликаты, временные границы, распределения и сверку "
            "агрегатов с источником. Результат записи дополнительно проверяют по числу "
            "строк и контрольной выборке."
        ),
    ),
    Intent(
        name="experiment",
        title="Гипотеза и эксперимент",
        patterns=(
            "как проверить гипотезу",
            "что важно в ab тесте",
            "как оценить эффект эксперимента",
        ),
        keywords={
            "гипотез*": 2.0,
            "эксперимент*": 2.0,
            "ab": 1.8,
            "a b": 1.8,
            "контрол*": 1.2,
            "эффект*": 1.2,
            "mde": 1.5,
            "тест*": 0.8,
            "причин*": 0.9,
        },
        answer=(
            "Для проверки гипотезы заранее фиксируют основную метрику, единицу "
            "рандомизации и анализа, контроль, окно наблюдения и минимально заметный "
            "эффект. После запуска проверяют баланс групп, качество логирования и "
            "доверительный интервал, а не только знак изменения."
        ),
    ),
    Intent(
        name="reporting",
        title="Аналитический отчёт",
        patterns=(
            "как оформить аналитический вывод",
            "что включить в отчет",
            "как представить результаты анализа",
        ),
        keywords={
            "отчет*": 1.9,
            "вывод*": 1.7,
            "результат*": 0.9,
            "рекомендац*": 1.2,
            "график*": 0.8,
            "документ*": 0.8,
            "презент*": 0.8,
            "оформ*": 1.0,
            "резюм*": 1.7,
            "итог*": 1.5,
            "исследован*": 0.6,
            "руководител*": 0.6,
        },
        answer=(
            "Хороший аналитический отчёт начинается с ответа на бизнес-вопрос. Затем "
            "идут подтверждающие факты, методика, ограничения и конкретные действия. "
            "Факты, интерпретации и гипотезы нужно маркировать отдельно; ключевые "
            "числа — сопровождать источником и проверкой."
        ),
    ),
    Intent(
        name="help",
        title="Возможности бота",
        patterns=(
            "что ты умеешь",
            "как пользоваться ботом",
            "с чем ты можешь помочь",
        ),
        keywords={
            "умееш*": 2.0,
            "помо*": 1.5,
            "возможност*": 1.8,
            "пользоват*": 1.0,
            "вопрос*": 0.8,
            "бот*": 0.8,
            "поддерж*": 1.3,
            "доступн*": 1.0,
            "список": 0.5,
            "тем*": 0.6,
        },
        answer=(
            "Я распознаю темы по ключевым словам и объясняю базовые понятия аналитики "
            "маркетинга Поиска. Я не подключена к внутренним системам, не показываю "
            "реальные корпоративные данные и не заменяю проверку первичных источников."
        ),
    ),
)


def normalize_text(text: str) -> str:
    """Normalize Russian text for deterministic keyword matching."""
    lowered = text.casefold().replace("ё", "е")
    cleaned = re.sub(r"[^a-zа-я0-9]+", " ", lowered, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _keyword_matches(keyword: str, normalized: str, tokens: set[str]) -> bool:
    is_prefix = keyword.strip().endswith("*")
    keyword = normalize_text(keyword.rstrip("*"))
    if not keyword:
        return False
    if is_prefix:
        if " " in keyword:
            parts = keyword.split()
            expression = r"\b" + r"\s+".join(
                [*(re.escape(part) for part in parts[:-1]), re.escape(parts[-1]) + r"\w*"]
            )
            return re.search(expression, normalized) is not None
        return any(token.startswith(keyword) for token in tokens)
    if " " in keyword:
        return keyword in normalized
    return keyword in tokens


def _score_intent(intent: Intent, normalized: str, tokens: set[str]) -> tuple[float, list[str]]:
    score = 0.0
    matches: list[str] = []
    for pattern in intent.patterns:
        normalized_pattern = normalize_text(pattern)
        if normalized_pattern and normalized_pattern in normalized:
            score += 3.5 + 0.25 * len(normalized_pattern.split())
            matches.append(pattern)
    for keyword, weight in intent.keywords.items():
        if _keyword_matches(keyword, normalized, tokens):
            score += weight
            matches.append(keyword)
    return score, matches


def classify(text: str, intents: Iterable[Intent] = INTENTS) -> ChatResponse:
    """Classify a request and return an explainable response."""
    normalized = normalize_text(text)
    if not normalized:
        return ChatResponse("fallback", "Нужен вопрос", 0.0, FALLBACK_ANSWER, (), ())

    tokens = set(normalized.split())
    ranked: list[tuple[float, Intent, list[str]]] = []
    for intent in intents:
        score, matches = _score_intent(intent, normalized, tokens)
        ranked.append((score, intent, matches))
    ranked.sort(key=lambda item: (-item[0], item[1].name))

    top_score, top_intent, top_matches = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if top_score < 1.1:
        return ChatResponse("fallback", "Тема не определена", 0.0, FALLBACK_ANSWER, (), ())

    margin = max(0.0, top_score - second_score)
    confidence = min(0.99, 0.42 + 0.075 * top_score + 0.045 * margin)
    alternatives = tuple(item[1].name for item in ranked[1:3] if item[0] > 0)
    return ChatResponse(
        intent=top_intent.name,
        title=top_intent.title,
        confidence=round(confidence, 3),
        answer=top_intent.answer,
        matched_keywords=tuple(dict.fromkeys(top_matches)),
        alternatives=alternatives,
    )


def available_topics() -> list[dict[str, str]]:
    return [{"intent": intent.name, "title": intent.title} for intent in INTENTS]
