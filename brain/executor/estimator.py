"""Оценка показателей канала по факту и детекция смены режима.

Масштаб задачи: при 1,2 млн ₽ на 21 день и восьми каналах на канал приходится
около 300 ₽ в час, то есть порядка тысячи показов и доли конверсии. Наивная
оценка на таких числах это шум, поэтому:

- CTR и CVR сглаживаются бета-биномиально с приором из кривой каталога
  (стандартная практика автобюджетов: приор для холодного старта);
- смена режима ловится сравнением отдачи на рубль за последние сутки с
  отдачей за предыдущие трое суток. Порог падения взят из практики
  мониторинга кампаний: падение CTR на 30–40 % считается аномалией
  (Improvado, Campaign Monitoring & Anomaly Detection Guide). Сравнение
  суток с сутками нечувствительно к медленному дрейфу усталости и к
  суточному профилю, а требование минимального числа кликов защищает от
  пуассоновского шума маленьких каналов. Классический CUSUM (Page, 1954)
  на часовых остатках проверен и отвергнут: часовой шум мира автокоррелирован
  и структурно расходится с планом, что дало 2–4 ложные тревоги на кампанию
  (docs/decisions.md);
- срабатывание детектора обесценивает накопленную статистику канала, чтобы
  оценка развернулась за часы, а не за неделю (грабли №4 в docs/mle2_plan.md).

Все константы в ``config/controller.yaml`` с происхождением.
"""

from collections import deque
from dataclasses import dataclass, field

from brain.config import (
    DETECTOR_BASELINE_HOURS,
    DETECTOR_CONFIRM_HOURS,
    DETECTOR_DROP_THRESHOLD,
    DETECTOR_MIN_CLICKS,
    DETECTOR_WINDOW_HOURS,
    DISCOUNT_ON_SHOCK,
    SHOCK_STATUS_HOURS,
)


@dataclass
class RateEstimator:
    """Бета-биномиальная оценка одной ставки (CTR или CVR)."""

    prior_rate: float
    prior_weight: float
    successes: float = 0.0
    trials: float = 0.0

    def update(self, successes: float, trials: float) -> None:
        self.successes += successes
        self.trials += trials

    def discount(self, keep: float = DISCOUNT_ON_SHOCK) -> None:
        self.successes *= keep
        self.trials *= keep

    @property
    def value(self) -> float:
        return (self.successes + self.prior_weight * self.prior_rate) / (self.trials + self.prior_weight)


@dataclass
class DropDetector:
    """Отдача за последние сутки против отдачи за предыдущие трое суток."""

    window_hours: int = DETECTOR_WINDOW_HOURS
    baseline_hours: int = DETECTOR_BASELINE_HOURS
    drop_threshold: float = DETECTOR_DROP_THRESHOLD
    min_clicks: float = DETECTOR_MIN_CLICKS
    confirm_hours: int = DETECTOR_CONFIRM_HOURS
    history: deque = field(default_factory=deque)  # (clicks, spend, expected_rate) по часам
    triggered_at: int | None = None
    last_ratio: float | None = None
    below_since: int | None = None

    def update(self, clicks: float, spend: float, expected_rate: float, hour: int) -> bool:
        self.history.append((clicks, spend, expected_rate))
        needed = self.window_hours + self.baseline_hours
        while len(self.history) > needed:
            self.history.popleft()
        if len(self.history) < needed:
            return False
        rows = list(self.history)
        recent = rows[-self.window_hours :]
        base = rows[: -self.window_hours]
        recent_clicks = sum(c for c, _, _ in recent)
        recent_spend = sum(s for _, s, _ in recent)
        base_clicks = sum(c for c, _, _ in base)
        base_spend = sum(s for _, s, _ in base)
        if recent_clicks < self.min_clicks or base_clicks < self.min_clicks or recent_spend <= 0 or base_spend <= 0:
            self.below_since = None
            return False
        actual_ratio = (recent_clicks / recent_spend) / (base_clicks / base_spend)
        # ожидаемое снижение по плану: усталость аудитории это не шок, а норма
        expected_recent = sum(e * s for _, s, e in recent) / recent_spend
        expected_base = sum(e * s for _, s, e in base) / base_spend
        expected_ratio = expected_recent / expected_base if expected_base > 0 else 1.0
        ratio = actual_ratio / expected_ratio if expected_ratio > 0 else actual_ratio
        self.last_ratio = ratio
        if ratio < 1 - self.drop_threshold:
            if self.below_since is None:
                self.below_since = hour
            if hour - self.below_since >= self.confirm_hours:
                self.triggered_at = hour
                self.history.clear()  # новый режим: базу строим заново
                self.below_since = None
                return True
        else:
            self.below_since = None
        return False


@dataclass
class ChannelEstimate:
    """Всё, что исполнитель знает о канале по факту."""

    channel_id: str
    ctr: RateEstimator
    cvr: RateEstimator
    plan_clicks_per_rub_by_day: list[float]
    ecpm_ewma: float | None = None
    detector: DropDetector = field(default_factory=DropDetector)
    cum_spend: float = 0.0
    cum_impressions: float = 0.0
    cum_clicks: float = 0.0
    cum_conversions: float = 0.0
    cum_reach: float = 0.0
    hours_without_delivery: int = 0
    shock_active: bool = False
    shock_hour: int | None = None

    def observe(self, obs, kpi: str, hour: int, alpha: float = 0.2) -> bool:
        """Обновляет оценки часовым наблюдением; True, если детектор сработал."""
        if self.shock_active and self.shock_hour is not None and hour - self.shock_hour >= SHOCK_STATUS_HOURS:
            self.shock_active = False  # новый режим принят как норма, статус «пожар» снимается
        self.cum_spend += obs.spend
        self.cum_impressions += obs.impressions
        self.cum_clicks += obs.clicks
        self.cum_conversions += obs.conversions
        self.cum_reach += obs.unique_reach
        self.ctr.update(obs.clicks, obs.impressions)
        self.cvr.update(obs.conversions, obs.clicks)
        if obs.impressions > 0:
            self.ecpm_ewma = obs.ecpm if self.ecpm_ewma is None else (1 - alpha) * self.ecpm_ewma + alpha * obs.ecpm
        self.hours_without_delivery = 0 if obs.impressions > 0 else self.hours_without_delivery + 1

        # Сигнал детектора: клики на рубль. Конверсии слишком редки для суточного
        # сигнала, а клики идут первыми при любом из трёх шоков: подорожание
        # (меньше показов на рубль), падение CTR, сжатие ёмкости.
        signal = float(obs.clicks) if kpi != "reach" else float(obs.unique_reach)
        day = min(max(hour - 1, 0) // 24, len(self.plan_clicks_per_rub_by_day) - 1)
        expected_rate = self.plan_clicks_per_rub_by_day[day] if self.plan_clicks_per_rub_by_day else 1.0
        fired = self.detector.update(signal, obs.spend, expected_rate, hour)
        if fired:
            self.shock_active = True
            self.shock_hour = hour
            self.ctr.discount()
            self.cvr.discount()
        return fired

    @property
    def observed_ecpm(self) -> float | None:
        return self.ecpm_ewma
