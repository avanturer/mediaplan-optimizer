"""Публичные допущения, которыми пользуется мозг.

Из ``config/assumptions.yaml`` мозг читает только то, что рекламодатель знал
бы и без симулятора: типичную усталость от частоты и во сколько раз охват
кампании превышает дневную аудиторию. Диапазоны, из которых мир выбирает
истинные значения, мозг не использует: он берёт середину как бенчмарк.
"""

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "assumptions.yaml"


@lru_cache(maxsize=1)
def _load() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fatigue_delta() -> float:
    return float(_load()["frequency_fatigue"]["delta"]["value"])


def campaign_audience_multiplier() -> float:
    return float(_load()["audience_scale"]["campaign_audience_multiplier"]["value"])
