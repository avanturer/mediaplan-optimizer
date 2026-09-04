"""Метрики качества исполнения.

Прямая метрика кейса: MAPE накопительных spend и KPI относительно плана к
концу кампании. Дополнительно, по контракту мира и обзору пейсинга:
WAPE (устойчив около нуля), конечное отклонение, unsmoothness index
(Smart Pacing) и коэффициент вариации управляющего сигнала (BHC).
"""

import numpy as np

from brain.config import MAPE_WARMUP_SHARE as WARMUP_SHARE


def mape(plan: np.ndarray, fact: np.ndarray) -> float:
    n = len(plan)
    start = int(n * WARMUP_SHARE)
    p, f = plan[start:], fact[start:]
    mask = p > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(f[mask] - p[mask]) / p[mask]))


def wape(plan: np.ndarray, fact: np.ndarray) -> float:
    denom = float(np.sum(np.abs(plan)))
    return float(np.sum(np.abs(fact - plan)) / denom) if denom > 0 else 0.0


def final_deviation(plan: np.ndarray, fact: np.ndarray) -> float:
    return float(abs(fact[-1] - plan[-1]) / plan[-1]) if plan[-1] > 0 else 0.0


def unsmoothness(plan_cum: np.ndarray, fact_cum: np.ndarray) -> float:
    """Средняя нормированная разница часового факта и часового плана."""
    plan_hourly = np.diff(np.concatenate([[0.0], plan_cum]))
    fact_hourly = np.diff(np.concatenate([[0.0], fact_cum]))
    mean_plan = float(np.mean(plan_hourly))
    return float(np.mean(np.abs(fact_hourly - plan_hourly)) / mean_plan) if mean_plan > 0 else 0.0


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    return float(arr.std() / mean) if mean > 0 else 0.0
