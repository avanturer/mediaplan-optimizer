"""Контракт симулятора: reset / step / inject_shock.

Форма взята из контракта модели мира (docs/world/WORLD_MODEL.md, §8–§10):
действие это вектор часовых лимитов расхода по каналам, наблюдение это
агрегаты завершённого часа. Ставки и множители в MVP отсутствуют; если
понадобятся, добавляются версионированным расширением, а не переопределением
смысла ``spend_caps``.

Расширение относительно контракта мира: ``inject_shock`` (шок задаётся из
интерфейса во время прогона, требование презентации кейса).
"""

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

API_VERSION = "1.1"


class SeedBundle(BaseModel):
    """Три зерна, которые контракт мира требует различать (§9).

    ``catalog_seed`` порождает публичный каталог, ``world_seed`` истинные
    параметры эпизода, ``noise_seed`` почасовую экзогенную случайность.
    Для честного сравнения стратегий меняется только policy.
    """

    catalog_seed: int = 0
    world_seed: int = 0
    noise_seed: int = 0


class ShockParameter(StrEnum):
    ECPM = "ecpm"
    CTR = "ctr"
    CVR = "cvr"
    INVENTORY = "inventory"
    DEMAND = "demand"
    PAUSE = "pause"
    FRAUD = "fraud"
    SMS_WEEKLY_LIMIT = "sms_weekly_limit"


class ShockEvent(BaseModel):
    """Декларативное событие сценария (контракт мира, §10)."""

    model_config = ConfigDict(allow_inf_nan=False)

    start_hour: int = Field(ge=0)
    duration_hours: int | None = Field(default=None, ge=1, description="None = до конца кампании")
    target_channels: list[str] = Field(min_length=1)
    parameter: ShockParameter
    multiplier: float = Field(gt=0, description="для pause игнорируется")
    recovery: str = Field(default="none", pattern="^(none|linear)$")

    @model_validator(mode="after")
    def validate_limit(self) -> "ShockEvent":
        if self.parameter == ShockParameter.SMS_WEEKLY_LIMIT and self.multiplier > 1:
            raise ValueError("sms_weekly_limit задаётся долей обычной недельной квоты (0, 1]")
        return self

    def factor_at(self, hour: int) -> float | None:
        """Множитель параметра в час ``hour`` или None, если событие не активно."""
        if hour < self.start_hour:
            return None
        if self.duration_hours is None:
            return self.multiplier
        end = self.start_hour + self.duration_hours
        if hour < end:
            return self.multiplier
        if self.recovery == "linear":
            # линейное восстановление к 1.0 за ту же длительность
            tail = hour - end
            if tail < self.duration_hours:
                share = 1 - tail / self.duration_hours
                return 1 + (self.multiplier - 1) * share
        return None


class Scenario(BaseModel):
    scenario_id: str
    events: list[ShockEvent] = Field(default_factory=list)


class Action(BaseModel):
    """Лимиты расхода на следующий час по каждому активному каналу, в рублях."""

    spend_caps: dict[str, float]

    @field_validator("spend_caps")
    @classmethod
    def caps_are_finite_and_nonnegative(cls, caps: dict[str, float]) -> dict[str, float]:
        for channel_id, cap in caps.items():
            if not math.isfinite(cap) or cap < 0:
                raise ValueError(f"лимит канала {channel_id} должен быть конечным и неотрицательным")
        return caps

    @property
    def total(self) -> float:
        return sum(self.spend_caps.values())


class ChannelObservation(BaseModel):
    """Факт по каналу за час. Состав полей задан кейсом дословно."""

    requests: int = Field(ge=0)
    impressions: int = Field(ge=0)
    unique_reach: int = Field(ge=0, description="новые уникальные за час, как «охват» у МТС DSP")
    clicks: int = Field(ge=0)
    conversions: int = Field(ge=0)
    spend: float = Field(ge=0)
    ecpm: float = Field(ge=0)
    fraud_share: float = Field(default=0.0, ge=0, le=1, description="доля показов, отмеченных измерителем как подозрительные; при отсутствии показов 0")
    verified_impressions: int = Field(default=0, ge=0, description="показы, не отмеченные измерителем; не oracle human traffic")

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions else 0.0

    @property
    def cvr(self) -> float:
        return self.conversions / self.clicks if self.clicks else 0.0


class Observation(BaseModel):
    hour: int = Field(ge=0, description="индекс завершённого часа от старта кампании")
    by_channel: dict[str, ChannelObservation]
    deduplicated_reach: int | None = Field(default=None, ge=0, description="новые уникальные кампании за час, с учётом пересечений и предыдущих часов")

    @property
    def total_spend(self) -> float:
        return sum(c.spend for c in self.by_channel.values())

    @property
    def total_clicks(self) -> int:
        return sum(c.clicks for c in self.by_channel.values())

    @property
    def total_conversions(self) -> int:
        return sum(c.conversions for c in self.by_channel.values())

    @property
    def total_impressions(self) -> int:
        return sum(c.impressions for c in self.by_channel.values())

    @property
    def total_reach(self) -> int:
        if self.deduplicated_reach is not None:
            return self.deduplicated_reach
        return sum(c.unique_reach for c in self.by_channel.values())


class StepMetrics(BaseModel):
    """Безопасные накопительные итоги, которые симулятор может отдать вместе с наблюдением."""

    cumulative_spend: float = Field(ge=0)
    cumulative_clicks: int = Field(ge=0)
    cumulative_conversions: int = Field(ge=0)
    remaining_budget: float = Field(ge=0)
    cumulative_reach: int = Field(default=0, ge=0)


class StepInfo(BaseModel):
    api_version: str = API_VERSION
    episode_id: str
    scenario_id: str
    applied_constraints: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)
    terminated_reason: str | None = None
    config_hash: str = ""
