"""Медиаплан: то, что медиапланер утверждает и за чем потом следит исполнитель.

План несёт плановую траекторию с коридором. Коридор нужен продукту
(сценарий С1: «коридор вместо точки», С4: статусы «в норме / наблюдаем»), а его
ширина берётся из неопределённости самого каталога, не выдумывается.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from contracts.brief import Brief


class ChannelAllocation(BaseModel):
    """Строка медиаплана: сколько денег в канал и что за них ожидаем."""

    channel_id: str
    display_name: str
    budget_rub: float = Field(ge=0)

    impressions: float = Field(ge=0)
    unique_reach: float = Field(ge=0)
    clicks: float = Field(ge=0)
    conversions: float = Field(ge=0)

    ctr: float = Field(ge=0, le=1)
    cvr: float = Field(ge=0, le=1)
    vtr: float | None = Field(default=None, ge=0, le=1)
    cpm_rub: float = Field(ge=0)
    cpc_rub: float | None = Field(default=None, ge=0)
    cpa_rub: float | None = Field(default=None, ge=0)
    frequency: float = Field(ge=0, description="среднее число контактов на охваченного")
    capacity_utilization: float = Field(ge=0, le=1, description="доля ёмкости канала, которую выкупаем")
    marginal_cost_per_1000_kpi_rub: float | None = Field(
        default=None,
        ge=0,
        description="«цена следующей тысячи» KPI в этом канале при текущем распределении (С1)",
    )
    expected_daily_clicks_per_rub: list[float] = Field(
        default_factory=list,
        description="плановая отдача по дням с учётом усталости; ожидание для детектора исполнителя",
    )
    locked: bool = False


class CalendarCell(BaseModel):
    day: int = Field(ge=1)
    channel_id: str
    budget_rub: float = Field(ge=0)


class TrajectoryPoint(BaseModel):
    """Точка плановой траектории. Именно с ней исполнитель сверяет факт."""

    hour: int = Field(ge=0, description="накопительно к концу этого часа (1..H)")
    cum_spend_rub: float = Field(ge=0)
    cum_impressions: float = Field(ge=0)
    cum_clicks: float = Field(ge=0)
    cum_conversions: float = Field(ge=0)
    cum_reach: float = Field(ge=0)
    band_low_spend_rub: float = Field(ge=0)
    band_high_spend_rub: float = Field(ge=0)
    band_low_kpi: float = Field(ge=0)
    band_high_kpi: float = Field(ge=0)
    by_channel_cum_spend_rub: dict[str, float] = Field(default_factory=dict)


class Forecast(BaseModel):
    kpi_name: str
    p10: float
    p50: float
    p90: float
    probability_of_target: float | None = Field(default=None, ge=0, le=1)


class BindingConstraint(StrEnum):
    CAPACITY = "capacity"
    HORIZON = "horizon"
    ECONOMICS = "economics"
    CHANNEL_SET = "channel_set"


class BriefSuggestion(BaseModel):
    """Готовая альтернатива с посчитанным результатом: кнопка, а не совет."""

    description: str
    changed_field: str
    suggested_value: float | list[str] | None
    expected_kpi: float
    expected_budget_rub: float


class Infeasibility(BaseModel):
    binding_constraint: BindingConstraint
    explanation: str
    max_achievable: float
    suggestions: list[BriefSuggestion] = Field(default_factory=list)


class MediaPlan(BaseModel):
    """Утверждаемый артефакт планирования."""

    plan_id: str
    ml_model_id: str | None = None
    catalog_id: str
    brief: Brief
    kpi_name: str

    allocations: list[ChannelAllocation] = Field(default_factory=list)
    calendar: list[CalendarCell] = Field(default_factory=list)
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    hourly_caps: list[dict[str, float]] = Field(
        default_factory=list,
        description="плановые часовые лимиты по каналам; это и есть «замороженное» исполнение",
    )
    forecast: Forecast | None = None
    corridor_rel: float = Field(default=0.0, ge=0, description="относительная полуширина коридора")
    explanation: list[str] = Field(default_factory=list)
    infeasibility: Infeasibility | None = None

    @property
    def is_feasible(self) -> bool:
        return self.infeasibility is None

    @property
    def total_budget_rub(self) -> float:
        return sum(a.budget_rub for a in self.allocations)

    @property
    def total_kpi(self) -> float:
        if not self.trajectory:
            return 0.0
        last = self.trajectory[-1]
        return {
            "clicks": last.cum_clicks,
            "conversions": last.cum_conversions,
            "reach": last.cum_reach,
        }[self.kpi_name]
