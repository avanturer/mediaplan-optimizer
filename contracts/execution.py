"""Решения сервиса исполнения и итог прогона.

Решение всегда объяснимо: рядом с числом лежит причина (требование кейса
«как перераспределялись средства»). Карточка предложения несёт две цены,
как просит продукт (сценарий С5): цену решения и цену бездействия.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ChannelStatus(StrEnum):
    ACTIVE = "active"
    FROZEN_CAPACITY = "frozen_capacity"
    FROZEN_ECONOMICS = "frozen_economics"
    PAUSED = "paused"


class TrackingStatus(StrEnum):
    """Статус кампании для экрана трафик-менеджера (С4)."""

    OK = "ok"  # внутри коридора
    WATCH = "watch"  # вышли за коридор
    FIRE = "fire"  # детектор увидел слом канала


class ChannelDecision(BaseModel):
    channel_id: str
    cap_rub: float = Field(ge=0, description="лимит расхода на следующий час")
    pacing_signal: float = Field(gt=0, description="множитель внутреннего контура λ")
    status: ChannelStatus = ChannelStatus.ACTIVE
    reason: str
    estimated_ctr: float = Field(ge=0, le=1)
    estimated_cvr: float = Field(ge=0, le=1)
    estimated_ecpm_rub: float = Field(ge=0)


class Proposal(BaseModel):
    """Карточка предложения (С5): что, куда, почему и чего стоит."""

    hour: int
    from_channel: str
    to_channel: str
    amount_rub: float = Field(ge=0)
    cause: str
    cost_of_decision: str = Field(description="цена решения словами, например «CPA вырастет на 3 %»")
    cost_of_inaction: str = Field(description="цена бездействия словами, например «недоберём 14 %»")
    cpa_delta_pct: float | None = None
    inaction_kpi_shortfall_pct: float | None = None
    applied_by: str = Field(pattern="^(system|human|pending)$")


class ExecutionDecision(BaseModel):
    """Полное решение на следующий час: куда деньги и почему."""

    hour: int = Field(ge=0, description="час, на который принято решение (следующий)")
    decisions: list[ChannelDecision]
    status: TrackingStatus = TrackingStatus.OK
    tracking_error_spend: float = Field(description="(план − факт) / план по накопленному расходу")
    tracking_error_kpi: float
    budget_remaining_rub: float = Field(ge=0)
    hours_remaining: int = Field(ge=0)
    shock_detected: list[str] = Field(default_factory=list)
    proposals: list[Proposal] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list, description="только то, что требует внимания")
    shadow_price: float | None = Field(default=None, description="оценка стоимости единицы KPI на остатке")

    @property
    def action(self) -> dict[str, float]:
        return {d.channel_id: d.cap_rub for d in self.decisions}


class HourRecord(BaseModel):
    """Одна строка почасового журнала для графиков и итога."""

    hour: int
    plan_cum_spend: float
    plan_cum_kpi: float
    fact_cum_spend: float
    fact_cum_kpi: float
    fact_by_channel_spend: dict[str, float]
    fact_by_channel: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="requests, impressions, unique_reach, clicks, conversions, spend, ecpm по каналу за час",
    )
    caps: dict[str, float]
    status: TrackingStatus
    events: list[str] = Field(default_factory=list)
    tracking_error_spend: float = 0.0
    tracking_error_kpi: float = 0.0
    reserve_rub: float = 0.0
    deduplicated_reach: int = 0


class RunSummary(BaseModel):
    """Итог одного прогона одной стратегии (С6)."""

    strategy: str
    scenario_id: str
    world_seed: int
    noise_seed: int
    kpi_name: str
    promised_spend: float
    promised_kpi: float
    actual_spend: float
    actual_kpi: float
    mape_spend: float = Field(description="MAPE накопительного расхода по часам")
    mape_kpi: float
    wape_spend: float
    wape_kpi: float
    final_deviation_spend: float = Field(description="|факт − план| / план в конце")
    final_deviation_kpi: float
    unsmoothness: float = Field(description="средняя нормированная разница часового факта и плана")
    lambda_cv: float = Field(description="коэффициент вариации управляющего сигнала")
    shock_hours: list[int] = Field(default_factory=list)
    detection_hours: dict[str, int] = Field(default_factory=dict, description="канал → час срабатывания детектора")
    human_requests: int = 0
    proposals: list[Proposal] = Field(default_factory=list)
    hours: list[HourRecord] = Field(default_factory=list)
    runtime_seconds: float = 0.0
