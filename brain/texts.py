"""Человеческие тексты мозга: имена KPI по-русски и числа с пробелами вместо запятых."""

KPI_LABELS = {"conversions": "конверсий", "clicks": "кликов", "reach": "охвата"}


def kpi_label(kpi: str) -> str:
    return KPI_LABELS.get(kpi, kpi)


def rub(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") + " ₽"


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")
