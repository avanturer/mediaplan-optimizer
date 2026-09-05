"""Константы мозга из ``config/controller.yaml``.

Каждое число в реестре имеет происхождение: публичный источник или
калибровка с тестом (tests/test_provenance.py). Здесь только чтение и
типизированный доступ; значений в коде нет.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "controller.yaml"


@lru_cache(maxsize=1)
def registry() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def const(section: str, name: str) -> Any:
    return registry()[section][name]["value"]


# планировщик
ALLOCATION_STEPS: int = int(const("planner", "allocation_steps"))
MODEL_GRID_SIZE: int = int(const("planner", "model_grid_size"))
CORRIDOR_SIGMA_DIVISOR: float = float(const("planner", "corridor_sigma_divisor"))
ECONOMICS_TAIL_MULTIPLIER: float = float(const("planner", "economics_tail_multiplier"))

# кривые
SATURATION_TOLERANCE: float = float(const("curves", "saturation_tolerance"))
PRIOR_IMPRESSIONS: float = float(const("curves", "prior_impressions"))
PRIOR_CLICKS: float = float(const("curves", "prior_clicks"))

# ретро-пробы
RETRO_LEVELS: tuple[float, ...] = tuple(float(x) for x in const("retro", "levels"))
RETRO_WORLD_SEEDS: tuple[int, ...] = tuple(int(x) for x in const("retro", "world_seeds"))
PUBLIC_CONTACTS_PER_USER: float = float(const("retro", "public_contacts_per_user"))

# детектор
DETECTOR_WINDOW_HOURS: int = int(const("detector", "window_hours"))
DETECTOR_BASELINE_HOURS: int = int(const("detector", "baseline_hours"))
DETECTOR_DROP_THRESHOLD: float = float(const("detector", "drop_threshold"))
DETECTOR_MIN_CLICKS: float = float(const("detector", "min_clicks"))
DETECTOR_CONFIRM_HOURS: int = int(const("detector", "confirm_hours"))
SHOCK_STATUS_HOURS: int = int(const("detector", "shock_status_hours"))
DETECTOR_RISE_THRESHOLD: float = float(const("detector", "rise_threshold"))
DETECTOR_KPI_WINDOW_HOURS: int = int(const("detector", "kpi_window_hours"))
DETECTOR_KPI_BASELINE_HOURS: int = int(const("detector", "kpi_baseline_hours"))
DETECTOR_Z_THRESHOLD: float = float(const("detector", "z_threshold"))
DETECTOR_MIN_EXPECTED_EVENTS: float = float(const("detector", "min_expected_events"))
DETECTOR_KPI_CONFIRM_HOURS: int = int(const("detector", "kpi_confirm_hours"))

# контроллер
DEAD_ZONE: float = float(const("controller", "dead_zone"))
ERROR_BANDS: tuple[tuple[float, float], ...] = tuple(
    (float(t), float(s)) for t, s in const("controller", "error_bands")
)
LAMBDA_MIN, LAMBDA_MAX = (float(x) for x in const("controller", "lambda_bounds"))
REPLAN_EVERY_HOURS: int = int(const("controller", "replan_every_hours"))
SHARE_RATE_LIMIT: float = float(const("controller", "share_rate_limit"))
PAUSE_AFTER_HOURS: int = int(const("controller", "pause_after_hours"))
PROBE_SHARE: float = float(const("controller", "probe_share"))
PROPOSAL_MIN_SHARE: float = float(const("controller", "proposal_min_share"))
REPLAN_GRID_SIZE: int = int(const("controller", "replan_grid_size"))
REPLAN_STEPS: int = int(const("controller", "replan_steps"))
PID_KP, PID_KI, PID_KD = (float(x) for x in const("controller", "pid_gains"))
RESERVE_WARMUP_SHARE: float = float(const("controller", "reserve_warmup_share"))
CARD_MIN_INTERVAL_HOURS: int = int(const("controller", "card_min_interval_hours"))
RESERVE_STEP_SHARE: float = float(const("controller", "reserve_step_share"))

# метрики
MAPE_WARMUP_SHARE: float = float(const("metrics", "mape_warmup_share"))
