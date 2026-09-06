"""Оценка показателей канала по факту и детекция смены режима.

Масштаб задачи: при 1,2 млн ₽ на 21 день и восьми каналах на канал приходится
около 300 ₽ в час, то есть порядка тысячи показов и доли конверсии. Наивная
оценка на таких числах это шум, поэтому:

- CTR и CVR сглаживаются бета-биномиально с приором из кривой каталога
  (стандартная практика автобюджетов: приор для холодного старта);
- смена режима ловится сравнением отдачи на рубль за последние сутки с
  отдачей за предыдущие трое суток. Порог падения взят из практики
  мониторинга кампаний: падение CTR на 30–40 % считается аномалией
  (Improvado, Campaign Monitoring & Anomaly Detection Guide); к порогу
  величины добавлен тест значимости на 3σ (условный биномиальный тест двух
  пуассоновских счётчиков), поэтому детектор сам подстраивается под шум мира. Сравнение
  суток с сутками нечувствительно к медленному дрейфу усталости и к
  суточному профилю, а требование минимального числа кликов защищает от
  пуассоновского шума маленьких каналов. Классический CUSUM (Page, 1954)
  на часовых остатках проверен и отвергнут: часовой шум мира автокоррелирован
  и структурно расходится с планом, что дало 2–4 ложные тревоги на кампанию
  (docs/decisions.md);
- срабатывание детектора заменяет накопленную статистику канала последним
  окном, в котором слом уже виден: оценка разворачивается сразу, а не за
  неделю (грабли №4 в docs/optimizer_plan.md);
- детектор двусторонний: скачок отдачи вверх при прежней цене тоже аномалия
  (типичная картина фрода: клики растут, конверсии нет), такой канал не
  получает денег за «улучшение», пока оно не подтвердится конверсиями;
- второй детектор смотрит на KPI кейса, конверсии на рубль: событий мало,
  поэтому окна длиннее (трое суток против пяти) и подтверждение сутки.

Все константы в ``config/controller.yaml`` с происхождением.
"""

from collections import deque
from dataclasses import dataclass, field

from brain.config import (
    DETECTOR_BASELINE_HOURS,
    DETECTOR_CONFIRM_HOURS,
    DETECTOR_DROP_THRESHOLD,
    DETECTOR_KPI_BASELINE_HOURS,
    DETECTOR_KPI_CONFIRM_HOURS,
    DETECTOR_KPI_WINDOW_HOURS,
    DETECTOR_MIN_CLICKS,
    DETECTOR_MIN_EXPECTED_EVENTS,
    DETECTOR_RISE_THRESHOLD,
    DETECTOR_WINDOW_HOURS,
    DETECTOR_Z_THRESHOLD,
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

    def reset(self) -> None:
        """Забыть накопленное: после слома старый режим ничего не говорит о новом."""
        self.successes = 0.0
        self.trials = 0.0

    @property
    def value(self) -> float:
        return (self.successes + self.prior_weight * self.prior_rate) / (self.trials + self.prior_weight)


@dataclass
class DropDetector:
    """Отдача за окно против отдачи за базу до него; ловит падение и, если задано, рост."""

    window_hours: int = DETECTOR_WINDOW_HOURS
    baseline_hours: int = DETECTOR_BASELINE_HOURS
    drop_threshold: float = DETECTOR_DROP_THRESHOLD
    rise_threshold: float | None = DETECTOR_RISE_THRESHOLD  # None = рост не считается аномалией
    min_events: float = DETECTOR_MIN_CLICKS
    confirm_hours: int = DETECTOR_CONFIRM_HOURS
    z_threshold: float | None = DETECTOR_Z_THRESHOLD  # падение должно быть ещё и значимым на этом уровне σ
    history: deque = field(default_factory=deque)  # (события, расход, ожидаемая отдача, показы) по часам
    triggered_at: int | None = None
    last_ratio: float | None = None
    below_since: int | None = None
    above_since: int | None = None
    base_snapshot: list | None = None  # база, замороженная в час первого подозрения: иначе она впитывает новый режим

    def update(self, events: float, spend: float, expected_rate: float, hour: int, impressions: float = 0.0) -> str | None:
        """Возвращает «drop», «rise» или None."""
        self.history.append((events, spend, expected_rate, impressions))
        needed = self.window_hours + self.baseline_hours
        while len(self.history) > needed:
            self.history.popleft()
        if len(self.history) < needed:
            return None
        rows = list(self.history)
        recent = rows[-self.window_hours :]
        base = self.base_snapshot if self.base_snapshot is not None else rows[: -self.window_hours]
        recent_events = sum(c for c, _, _, _ in recent)
        recent_spend = sum(s for _, s, _, _ in recent)
        base_events = sum(c for c, _, _, _ in base)
        base_spend = sum(s for _, s, _, _ in base)
        if recent_events < self.min_events or base_events < self.min_events or recent_spend <= 0 or base_spend <= 0:
            self._clear_suspicion()
            return None
        actual_ratio = (recent_events / recent_spend) / (base_events / base_spend)
        # ожидаемое снижение по плану: усталость аудитории это не шок, а норма
        expected_recent = sum(e * s for _, s, e, _ in recent) / recent_spend
        expected_base = sum(e * s for _, s, e, _ in base) / base_spend
        expected_ratio = expected_recent / expected_base if expected_base > 0 else 1.0
        ratio = actual_ratio / expected_ratio if expected_ratio > 0 else actual_ratio
        self.last_ratio = ratio
        significant = True
        if self.z_threshold is not None:
            # два пуассоновских счётчика: при неизменной отдаче доля событий окна среди всех
            # событий биномиальна с p0 = ожидаемая доля окна (условный тест двух счётчиков);
            # нормальное приближение допустимо при ожидаемых событиях не меньше 10 с каждой стороны
            total = recent_events + base_events
            p0 = expected_recent * recent_spend / (expected_recent * recent_spend + expected_base * base_spend)
            if total * p0 < DETECTOR_MIN_EXPECTED_EVENTS or total * (1 - p0) < DETECTOR_MIN_EXPECTED_EVENTS:
                self._clear_suspicion()
                return None
            sd = (total * p0 * (1 - p0)) ** 0.5
            z = (recent_events - total * p0) / sd if sd > 0 else 0.0
            significant = z <= -self.z_threshold
        if ratio < 1 - self.drop_threshold and significant:
            self.above_since = None
            if self.below_since is None:
                self.below_since = hour
                self.base_snapshot = list(base)
            if hour - self.below_since >= self.confirm_hours:
                self._fire(hour)
                return "drop"
            return None
        if self.rise_threshold is not None and self._ctr_jump(recent, base):
            self.below_since = None
            if self.above_since is None:
                self.above_since = hour
                self.base_snapshot = list(base)
            if hour - self.above_since >= self.confirm_hours:
                self._fire(hour)
                return "rise"
            return None
        self._clear_suspicion()
        return None

    def _clear_suspicion(self) -> None:
        self.below_since = self.above_since = None
        self.base_snapshot = None

    def _ctr_jump(self, recent, base) -> bool:
        """Рост CTR (клики на показ), а не кликов на рубль: цена и уровень расхода тут ни при чём.

        Порог по величине плюс значимость на 3σ: доля кликов окна среди всех кликов
        биномиальна с p0 = доля показов окна.
        """
        recent_clicks = sum(c for c, _, _, _ in recent)
        base_clicks = sum(c for c, _, _, _ in base)
        recent_imps = sum(i for _, _, _, i in recent)
        base_imps = sum(i for _, _, _, i in base)
        if recent_imps <= 0 or base_imps <= 0 or base_clicks <= 0:
            return False
        ratio = (recent_clicks / recent_imps) / (base_clicks / base_imps)
        if ratio <= 1 + self.rise_threshold:
            return False
        total = recent_clicks + base_clicks
        p0 = recent_imps / (recent_imps + base_imps)
        if total * p0 < DETECTOR_MIN_EXPECTED_EVENTS or total * (1 - p0) < DETECTOR_MIN_EXPECTED_EVENTS:
            return False
        sd = (total * p0 * (1 - p0)) ** 0.5
        return sd > 0 and (recent_clicks - total * p0) / sd >= DETECTOR_Z_THRESHOLD

    def _fire(self, hour: int) -> None:
        self.triggered_at = hour
        self.history.clear()  # новый режим: базу строим заново
        self._clear_suspicion()


def kpi_detector() -> DropDetector:
    """Детектор по конверсиям на рубль: окна длиннее, рост не считается аномалией.

    Событий мало, поэтому порог в процентах дополнен тестом значимости на 3σ:
    в маленьком канале он честно молчит, в крупном ловит падение CR за несколько суток.
    """
    return DropDetector(
        window_hours=DETECTOR_KPI_WINDOW_HOURS,
        baseline_hours=DETECTOR_KPI_BASELINE_HOURS,
        rise_threshold=None,
        min_events=1.0,
        confirm_hours=DETECTOR_KPI_CONFIRM_HOURS,
    )


@dataclass
class ChannelEstimate:
    """Всё, что исполнитель знает о канале по факту."""

    channel_id: str
    ctr: RateEstimator
    cvr: RateEstimator
    plan_clicks_per_rub_by_day: list[float]
    ecpm_ewma: float | None = None
    detector: DropDetector = field(default_factory=DropDetector)
    conversion_detector: DropDetector = field(default_factory=kpi_detector)
    cum_spend: float = 0.0
    cum_impressions: float = 0.0
    cum_clicks: float = 0.0
    cum_conversions: float = 0.0
    cum_reach: float = 0.0
    hours_without_delivery: int = 0
    shock_active: bool = False
    shock_hour: int | None = None
    last_event: str | None = None  # drop | rise | pause
    last_signal: str | None = None  # clicks | conversions: какой детектор сработал
    suspicious: bool = False  # рост CTR без подтверждения конверсиями: оценке CTR не верим вверх
    recent: deque = field(default_factory=deque)  # (показы, клики, конверсии) за последнее окно детектора

    def observe(self, obs, kpi: str, hour: int, alpha: float = 0.2, rate_scale: float = 1.0) -> bool:
        """Обновляет оценки часовым наблюдением; True, если сработал любой детектор."""
        if self.shock_active and self.shock_hour is not None and hour - self.shock_hour >= SHOCK_STATUS_HOURS:
            self.shock_active = False  # новый режим принят как норма, статус «пожар» снимается
            self.suspicious = False
        self.cum_spend += obs.spend
        self.cum_impressions += obs.impressions
        self.cum_clicks += obs.clicks
        self.cum_conversions += obs.conversions
        self.cum_reach += obs.unique_reach
        self.ctr.update(obs.clicks, obs.impressions)
        self.cvr.update(obs.conversions, obs.clicks)
        self.recent.append((obs.impressions, obs.clicks, obs.conversions))
        while len(self.recent) > self.detector.window_hours:
            self.recent.popleft()
        if obs.impressions > 0:
            self.ecpm_ewma = obs.ecpm if self.ecpm_ewma is None else (1 - alpha) * self.ecpm_ewma + alpha * obs.ecpm
        self.hours_without_delivery = 0 if obs.impressions > 0 else self.hours_without_delivery + 1

        # Первый сигнал: клики на рубль. Они идут первыми при подорожании, падении
        # CTR и сжатии ёмкости, и растут первыми при фроде. Второй сигнал: конверсии
        # на рубль, KPI кейса; ловит падение CR, которого клики не видят.
        day = min(max(hour - 1, 0) // 24, len(self.plan_clicks_per_rub_by_day) - 1)
        expected_rate = (self.plan_clicks_per_rub_by_day[day] if self.plan_clicks_per_rub_by_day else 1.0) * rate_scale
        if kpi == "reach":
            # Новый охват на рубль убывает с насыщением пула быстрее кликов, и сравнение с плановыми
            # кликами на рубль давало 5–7 ложных тревог за спокойную кампанию (ревью 06.09).
            # Для цели «охват» детектор по отдаче выключен: слом ловится паузой и расходом.
            event = None
        else:
            event = self.detector.update(float(obs.clicks), obs.spend, expected_rate, hour, impressions=float(obs.impressions))
        self.last_signal = "clicks" if event else None
        if event is None and kpi == "conversions":
            # плановая отдача по конверсиям пропорциональна кликам на рубль (CR плана постоянен по дням)
            event = self.conversion_detector.update(float(obs.conversions), obs.spend, expected_rate, hour)
            self.last_signal = "conversions" if event else None
        if event == "drop":
            self.shock_active, self.shock_hour, self.last_event = True, hour, "drop"
            # история заменяется последним окном детектора, в котором слом уже виден: если
            # оставить хоть часть старого режима, оценка откатится к каталогу и канал не будет
            # урезан, пока новый режим не наберёт статистику с нуля (проверено: 0,15 старого мало)
            self.ctr.reset()
            self.cvr.reset()
            imps = sum(i for i, _, _ in self.recent)
            clicks = sum(c for _, c, _ in self.recent)
            conv = sum(v for _, _, v in self.recent)
            self.ctr.update(clicks, imps)
            self.cvr.update(conv, clicks)
        elif event == "rise":
            self.shock_active, self.shock_hour, self.last_event = True, hour, "rise"
            self.suspicious = True
        return event is not None

    def mark_paused(self, hour: int) -> None:
        """Канал перестал отдавать показы: это слом, а не тишина."""
        self.shock_active, self.shock_hour, self.last_event = True, hour, "pause"

    @property
    def observed_ecpm(self) -> float | None:
        return self.ecpm_ewma
