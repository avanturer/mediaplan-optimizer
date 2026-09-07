"""Привилегированная конфигурация стенда. Не передаётся оптимизатору."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Competitor(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    competitor_id: str = Field(min_length=1)
    strength: float = Field(default=0.25, ge=0, le=2)
    channel_advantages: dict[str, float] = Field(default_factory=dict)
    volatility: float = Field(default=0.25, ge=0, le=1)
    start_hour: int = Field(default=0, ge=0)
    end_hour: int | None = Field(default=None, ge=1)

    @field_validator("channel_advantages")
    @classmethod
    def bounded_advantages(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not 0 <= x <= 3 for x in values.values()):
            raise ValueError("преимущества каналов должны быть в [0, 3]")
        return values

    @model_validator(mode="after")
    def ordered_hours(self) -> "Competitor":
        if self.end_hour is not None and self.end_hour <= self.start_hour:
            raise ValueError("end_hour должен быть позже start_hour")
        return self


class WorldSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # O_ij = |A_i ∩ A_j| / min(|A_i|, |A_j|). Только попарные общие пулы.
    default_overlap: float = Field(default=0.10, ge=0, le=1)
    overlap_matrix: dict[str, dict[str, float]] | None = None
    fraud_baseline: float = Field(default=0.03, ge=0, lt=1)
    bot_ctr: float = Field(default=0.20, ge=0, le=1)
    fraud_detection_rate: float = Field(default=0.95, ge=0, le=1)
    fraud_false_positive_rate: float = Field(default=0.001, ge=0, le=1)
    competitors: list[Competitor] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_competitors(self) -> "WorldSettings":
        ids = [c.competitor_id for c in self.competitors]
        if len(ids) != len(set(ids)):
            raise ValueError("competitor_id должны быть уникальными")
        return self
