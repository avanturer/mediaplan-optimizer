"""Бриф: вход планировщика. Две постановки кейса живут в одной модели."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from contracts.targeting import AudienceTargeting


class Objective(StrEnum):
    """Что максимизируем в постановке типа A."""

    MAX_CLICKS = "max_clicks"
    MAX_CONVERSIONS = "max_conversions"
    MAX_REACH = "max_reach"


class TargetKpi(StrEnum):
    """По какому показателю задана цель в постановке типа B."""

    REACH = "reach"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"


class Brief(BaseModel):
    """Тип A: задан бюджет, максимизируем KPI. Тип B: задан объём KPI, ищем бюджет.

    Ровно одно из двух заполнено. Отдельного поля «тип» нет, чтобы состояние
    не могло разъехаться.
    """

    budget_rub: float | None = Field(default=None, gt=0)
    target_kpi: TargetKpi | None = None
    target_value: float | None = Field(default=None, gt=0)

    objective: Objective = Objective.MAX_CONVERSIONS
    targeting: AudienceTargeting = Field(default_factory=AudienceTargeting)
    horizon_days: int = Field(ge=1, le=90)
    start_at: datetime = Field(default_factory=lambda: datetime(2026, 9, 14))
    channel_ids: list[str] = Field(min_length=1, description="набор каналов из пресета")

    max_cpa_rub: float | None = Field(default=None, gt=0, description="«средняя цена конверсии»")
    locked: dict[str, float] = Field(
        default_factory=dict,
        description="фиксированные бюджеты каналов (сценарий С1: человек двигает канал руками)",
    )
    automation_limit_rub: float | None = Field(
        default=None,
        ge=0,
        description="лимит полномочий автоматики за один ход (С3/С5); None = без лимита",
    )

    @model_validator(mode="after")
    def exactly_one_formulation(self) -> "Brief":
        has_budget = self.budget_rub is not None
        has_target = self.target_kpi is not None and self.target_value is not None
        if has_budget == has_target:
            raise ValueError(
                "укажите либо бюджет (тип A), либо цель и её объём (тип B), но не оба сразу"
            )
        unknown = set(self.locked) - set(self.channel_ids)
        if unknown:
            raise ValueError(f"зафиксированы каналы вне набора брифа: {sorted(unknown)}")
        return self

    @property
    def is_budget_constrained(self) -> bool:
        return self.budget_rub is not None

    @property
    def horizon_hours(self) -> int:
        return self.horizon_days * 24

    @property
    def kpi_name(self) -> str:
        """Имя основного KPI: для типа B из цели, для типа A из objective."""
        if self.target_kpi is not None:
            return self.target_kpi.value
        return {
            Objective.MAX_CLICKS: "clicks",
            Objective.MAX_CONVERSIONS: "conversions",
            Objective.MAX_REACH: "reach",
        }[self.objective]
