"""Чтение реестра бенчмарков и допущений.

Единственное место, где мир читает YAML. Всё, что здесь загружается, имеет
происхождение: ``config/benchmarks.yaml`` со ссылками на источники и
``config/assumptions.yaml`` с явно помеченными допущениями.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@lru_cache(maxsize=1)
def load_benchmarks() -> dict[str, Any]:
    with (CONFIG_DIR / "benchmarks.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_assumptions() -> dict[str, Any]:
    with (CONFIG_DIR / "assumptions.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def value_of(node: dict[str, Any]) -> float:
    """Достаёт ``value`` из узла вида ``{value: ..., range: [...]}``."""
    return float(node["value"])


def range_of(node: dict[str, Any]) -> tuple[float, float]:
    lo, hi = node["range"]
    return float(lo), float(hi)
