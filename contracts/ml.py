"""Переключатели и публичный журнал ML. Скрытого состояния мира здесь нет."""

from pydantic import BaseModel, ConfigDict


class MLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomaly_detection: bool = False
    response_curves: bool = False
    reach_correction: bool = False

    @property
    def enabled(self) -> bool:
        return any((self.anomaly_detection, self.response_curves, self.reach_correction))


class MLForecast(BaseModel):
    generated_at_hour: int
    forecast_for_hour: int
    predicted_kpi: float
    baseline_predicted_kpi: float
    predicted_reach: float
    additive_predicted_reach: float
    predicted_spend: float
    predicted_by_channel: dict[str, dict[str, float]]


class MLSignal(BaseModel):
    score: float | None = None
    threshold: float = 1.0
    alert: bool = False
    reason: str = "недостаточно истории"
