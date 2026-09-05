"""Распределение бюджета по предельной отдаче.

Для вогнутых сепарабельных кривых отклика оптимум описывается равенством
предельных отдач по каналам (Little & Lodish, «A Media Planning Calculus»,
1969; та же логика в аллокаторах Meridian и Robyn). Реализовано жадным
наливанием порциями: ёмкость входит естественно (у насыщенного канала
предельная отдача ноль), счёт занимает миллисекунды, а порядок наливания
сам является объяснением плана для медиапланера.
"""

from dataclasses import dataclass, field

import numpy as np

from brain.config import ALLOCATION_STEPS as STEPS
from brain.config import MODEL_GRID_SIZE as GRID_SIZE
from brain.curves import ResponseCurve

OUTCOME_KEYS = ("impressions", "reach", "clicks", "conversions", "spend")


@dataclass
class Outcome:
    """Результат канала за кампанию и его накопление по дням."""

    impressions: float
    reach: float
    clicks: float
    conversions: float
    spend: float
    daily_cum: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ChannelModel:
    """Функция «бюджет на кампанию → результат» одного канала с усталостью по дням.

    Кривая описывает день при частоте 1; накопление частоты и падение CTR
    делает модель по дням, ровно как мир делает по часам. Применять итоговую
    частоту ко всему объёму нельзя: в первый день частота равна единице.
    """

    channel_id: str
    curve: ResponseCurve
    days: int
    pool: float
    fatigue_delta: float
    grid_size: int = GRID_SIZE
    grid_budget: np.ndarray = field(default_factory=lambda: np.zeros(1))
    grid: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        max_total = self.curve.max_daily_spend * self.days
        self.grid_budget = np.linspace(0.0, max(max_total, 1.0), self.grid_size)
        rows = [self.simulate(b) for b in self.grid_budget]
        self.grid = {key: np.array([getattr(r, key) for r in rows]) for key in OUTCOME_KEYS}

    def simulate(self, total_budget: float) -> Outcome:
        daily = total_budget / self.days
        imps_day = self.curve.impressions_at(daily)
        spend_day = self.curve.effective_spend(daily)
        base_ctr, base_cvr = self.curve.rates_at(daily)
        cum = {key: np.zeros(self.days) for key in OUTCOME_KEYS}
        cum_imps = cum_reach = clicks = conv = spend = 0.0
        for d in range(self.days):
            new_reach = (self.pool - cum_reach) * (1 - np.exp(-imps_day / self.pool)) if self.pool > 0 else 0.0
            cum_reach += new_reach
            cum_imps += imps_day
            spend += spend_day
            freq = cum_imps / cum_reach if cum_reach > 0 else 1.0
            ctr = base_ctr / (1 + self.fatigue_delta * max(freq - 1, 0.0))
            day_clicks = imps_day * ctr
            clicks += day_clicks
            conv += day_clicks * base_cvr
            cum["impressions"][d], cum["reach"][d], cum["clicks"][d] = cum_imps, cum_reach, clicks
            cum["conversions"][d], cum["spend"][d] = conv, spend
        return Outcome(cum_imps, cum_reach, clicks, conv, spend, cum)

    def value(self, total_budget: float, kpi: str) -> float:
        return float(np.interp(total_budget, self.grid_budget, self.grid[kpi]))

    def outcome(self, total_budget: float) -> Outcome:
        """Точный пересчёт по дням для итоговой строки плана и траектории."""
        return self.simulate(min(max(total_budget, 0.0), self.max_budget))

    @property
    def max_budget(self) -> float:
        return float(self.grid_budget[-1])


def build_models(
    curves: dict[str, ResponseCurve],
    days: int,
    pools: dict[str, float],
    fatigue_delta: float,
    grid_size: int = GRID_SIZE,
) -> dict[str, ChannelModel]:
    return {
        cid: ChannelModel(
            channel_id=cid, curve=curve, days=days, pool=pools[cid], fatigue_delta=fatigue_delta, grid_size=grid_size
        )
        for cid, curve in curves.items()
    }


@dataclass
class AllocationResult:
    budgets: dict[str, float]
    unspent: float
    steps: list[str]
    frozen: dict[str, str]  # канал → причина заморозки
    marginal_cost_per_kpi: dict[str, float | None]


def allocate(
    models: dict[str, ChannelModel],
    budget: float,
    kpi: str,
    locked: dict[str, float] | None = None,
    max_cost_per_kpi: float | None = None,
    steps: int = STEPS,
    reach_model=None,
    prior_reach: dict[str, float] | None = None,
) -> AllocationResult:
    """Жадное наливание порциями ``budget / steps`` в канал с максимальным приростом KPI.

    ``locked`` фиксирует бюджеты каналов (человек двигает канал руками,
    остальные перераспределяются). ``max_cost_per_kpi`` замораживает канал,
    когда следующая порция стоит дороже допустимого (ограничение «средняя
    цена конверсии»).
    """
    locked = dict(locked or {})
    budgets = {cid: 0.0 for cid in models}
    for cid, value in locked.items():
        budgets[cid] = min(value, models[cid].max_budget)
    free_budget = budget - sum(budgets.values())
    eps = max(budget / steps, 1.0)
    frozen: dict[str, str] = {cid: "зафиксирован вручную" for cid in locked}
    order: list[str] = []
    explanation: list[str] = []
    joint_reach = reach_model if kpi == "reach" else None
    reach_values = {cid: model.value(budgets[cid], "reach") for cid, model in models.items()} if joint_reach else {}

    while free_budget >= eps * 0.5:
        best_cid, best_gain = None, 0.0
        current_reach = joint_reach.incremental(reach_values, prior_reach) if joint_reach else 0
        for cid, model in models.items():
            if cid in frozen:
                continue
            b = budgets[cid]
            if b + eps > model.max_budget:
                frozen[cid] = "ёмкость исчерпана"
                continue
            gain = model.value(b + eps, kpi) - model.value(b, kpi)
            if joint_reach:
                candidate = {**reach_values, cid: model.value(b + eps, "reach")}
                gain = joint_reach.incremental(candidate, prior_reach) - current_reach
            if max_cost_per_kpi is not None and gain > 0 and eps / gain > max_cost_per_kpi:
                frozen[cid] = f"следующая порция дороже лимита ({eps / gain:,.0f} ₽ за единицу KPI)"
                continue
            if gain > best_gain:
                best_cid, best_gain = cid, gain
        if best_cid is None:
            break
        if best_cid not in order:
            order.append(best_cid)
            explanation.append(
                f"шаг {len(order)}: {best_cid} получает бюджет, предельная цена "
                f"{eps / best_gain:,.0f} ₽ за единицу KPI"
            )
        budgets[best_cid] += eps
        if joint_reach:
            reach_values[best_cid] = models[best_cid].value(budgets[best_cid], "reach")
        free_budget -= eps

    for cid, reason in frozen.items():
        if cid not in locked:
            explanation.append(f"{cid}: остановлен, {reason}")

    marginal: dict[str, float | None] = {}
    for cid, model in models.items():
        b = budgets[cid]
        gain = model.value(min(b + eps, model.max_budget), kpi) - model.value(b, kpi)
        if joint_reach:
            candidate = {**reach_values, cid: model.value(min(b + eps, model.max_budget), "reach")}
            gain = joint_reach.incremental(candidate, prior_reach) - joint_reach.incremental(reach_values, prior_reach)
        marginal[cid] = eps / gain if gain > 1e-9 else None
    return AllocationResult(
        budgets=budgets,
        unspent=free_budget if free_budget > eps * 0.5 else 0.0,
        steps=explanation,
        frozen=frozen,
        marginal_cost_per_kpi=marginal,
    )
