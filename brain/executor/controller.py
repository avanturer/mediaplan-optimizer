"""Стратегии исполнения: static, proportional_pacing, pid и adaptive.

Все стратегии наследуют общую сверку факта с планом; отличаются только тем,
как считают лимиты следующего часа.

- ``static``: план исполняется буквально (baseline кейса «зафиксировать и не менять»).
- ``proportional_pacing``: остаток бюджета канала раскладывается по оставшимся
  часам пропорционально плановому профилю, формула Turn (ADKDD'13, 5–8) и
  Smart Pacing (KDD'15, формула 5). Перераспределения между каналами нет.
- ``pid``: плановый лимит канала умножается на PID-коррекцию по ошибке
  накопительного расхода (обзор Chen 2025; патент Adobe US10878448B1).
- ``adaptive``: два контура. Внешний раз в несколько часов перерешает
  распределение остатка тем же жадным алгоритмом на кривых, поправленных
  фактом (receding horizon); внутренний держит темп бакетизированным
  мультипликативным множителем (arXiv:2509.25429) с мёртвой зоной и
  ограничителями. Детектор CUSUM переключает оценки и вызывает перерешение.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from brain.assumptions import campaign_audience_multiplier, fatigue_delta
from brain.config import (
    CARD_MIN_INTERVAL_HOURS,
    DEAD_ZONE,
    ERROR_BANDS,
    LAMBDA_MAX,
    LAMBDA_MIN,
    PAUSE_AFTER_HOURS,
    PID_KD,
    PID_KI,
    PID_KP,
    PRIOR_CLICKS,
    PRIOR_IMPRESSIONS,
    PROBE_SHARE,
    PROPOSAL_MIN_SHARE,
    REPLAN_EVERY_HOURS,
    REPLAN_GRID_SIZE,
    REPLAN_STEPS,
    RESERVE_STEP_SHARE,
    RESERVE_WARMUP_SHARE,
    SHARE_RATE_LIMIT,
)
from brain.curves import CurvePoint, ResponseCurve
from brain.executor.estimator import ChannelEstimate, RateEstimator
from brain.planner.allocator import allocate as greedy_allocate
from brain.planner.allocator import build_models
from contracts import (
    ChannelDecision,
    ChannelStatus,
    ExecutionDecision,
    MediaPlan,
    Observation,
    Proposal,
    PublicCatalog,
    TrackingStatus,
)

HOURS_IN_WEEK = 168


def _kpi_of(obs_channel, kpi: str) -> float:
    return {"clicks": obs_channel.clicks, "conversions": obs_channel.conversions, "reach": obs_channel.unique_reach}[kpi]


@dataclass
class BaseExecutor:
    """Общее состояние: план, факт, оценки, статус. Подклассы задают ``_caps``."""

    plan: MediaPlan
    catalog: PublicCatalog
    curves: dict[str, ResponseCurve]
    total_budget: float
    name: str = "base"
    hour: int = 0  # сколько часов уже прошло
    fact_cum_spend: float = 0.0
    fact_cum_kpi: float = 0.0
    fact_cum_by_channel: dict[str, float] = field(default_factory=dict)
    estimates: dict[str, ChannelEstimate] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    detection_hours: dict[str, int] = field(default_factory=dict)
    last_caps: dict[str, float] = field(default_factory=dict)
    lambdas: list[float] = field(default_factory=list)
    human_requests: int = 0

    def __post_init__(self) -> None:
        self.channel_ids = [a.channel_id for a in self.plan.allocations]
        self.horizon = len(self.plan.trajectory)
        self.kpi = self.plan.kpi_name
        self.plan_caps = self.plan.hourly_caps
        self.plan_budget = {a.channel_id: a.budget_rub for a in self.plan.allocations}
        self.fact_cum_by_channel = {cid: 0.0 for cid in self.channel_ids}
        for alloc in self.plan.allocations:
            curve = self.curves[alloc.channel_id]
            self.estimates[alloc.channel_id] = ChannelEstimate(
                channel_id=alloc.channel_id,
                ctr=RateEstimator(curve.ctr, PRIOR_IMPRESSIONS),
                cvr=RateEstimator(curve.cvr, PRIOR_CLICKS),
                plan_clicks_per_rub_by_day=list(alloc.expected_daily_clicks_per_rub),
            )

    # ------------------------------------------------------------ наблюдение
    def observe(self, obs: Observation) -> list[str]:
        """Принимает факт завершённого часа; возвращает события, требующие внимания."""
        fired: list[str] = []
        for cid in self.channel_ids:
            ch = obs.by_channel[cid]
            self.fact_cum_by_channel[cid] += ch.spend
            self.fact_cum_spend += ch.spend
            self.fact_cum_kpi += _kpi_of(ch, self.kpi)
            if self.estimates[cid].observe(ch, self.kpi, obs.hour, rate_scale=self._rate_scale(cid, obs.hour - 1)):
                fired.append(cid)
                self.detection_hours.setdefault(cid, obs.hour)
        self.hour = obs.hour
        new_events = []
        for cid in fired:
            est = self.estimates[cid]
            if est.last_event == "rise":
                new_events.append(
                    f"час {obs.hour}: детектор: отдача канала {cid} выросла на {(est.detector.last_ratio or 1) - 1:.0%} при прежней цене: "
                    "похоже на фрод, рост не учитываем до подтверждения конверсиями"
                )
            else:
                what = "конверсии на рубль" if est.last_signal == "conversions" else "клики на рубль"
                new_events.append(f"час {obs.hour}: детектор: {what} в канале {cid} упали, оценки сброшены")
        for cid in self.channel_ids:
            est = self.estimates[cid]
            if est.hours_without_delivery == PAUSE_AFTER_HOURS and self.last_caps.get(cid, 0.0) > 0:
                # пауза канала это слом: попадает в детектор и поднимает статус, как и падение отдачи
                est.mark_paused(obs.hour)
                self.detection_hours.setdefault(cid, obs.hour)
                new_events.append(f"час {obs.hour}: канал {cid} не отдаёт показы {PAUSE_AFTER_HOURS} часа подряд")
        self.events.extend(new_events)
        self._after_observe(fired)
        return new_events

    def _after_observe(self, fired: list[str]) -> None:  # хук для adaptive
        return None

    # ---------------------------------------------------------------- решение
    def decide(self, remaining_budget: float) -> ExecutionDecision:
        h = self.hour  # следующий час имеет индекс h (0-based)
        hours_remaining = self.horizon - h
        caps = self._caps(h, remaining_budget) if hours_remaining > 0 else {cid: 0.0 for cid in self.channel_ids}
        caps = _clamp_total(caps, remaining_budget)
        self.last_caps = caps
        err_spend, err_kpi = self._tracking_errors(h)
        status = self._status(err_spend, err_kpi)
        decisions = [
            ChannelDecision(
                channel_id=cid,
                cap_rub=caps[cid],
                pacing_signal=self._lambda(),
                status=self._channel_status(cid, caps[cid]),
                reason=self._reason(cid, caps[cid], h),
                estimated_ctr=self.estimates[cid].ctr.value,
                estimated_cvr=self.estimates[cid].cvr.value,
                estimated_ecpm_rub=self.estimates[cid].observed_ecpm or self.catalog.by_id(cid).ecpm_mid,
            )
            for cid in self.channel_ids
        ]
        self.lambdas.append(self._lambda())
        return ExecutionDecision(
            hour=h,
            decisions=decisions,
            status=status,
            tracking_error_spend=err_spend,
            tracking_error_kpi=err_kpi,
            budget_remaining_rub=remaining_budget,
            hours_remaining=hours_remaining,
            shock_detected=[cid for cid, est in self.estimates.items() if est.shock_active],
            proposals=[p for p in self.proposals if p.hour == h],
            events=[e for e in self.events if e.startswith(f"час {h}:")],
            shadow_price=self._shadow_price(remaining_budget),
        )

    # ------------------------------------------------------------ служебное
    def _caps(self, h: int, remaining_budget: float) -> dict[str, float]:
        raise NotImplementedError

    def _lambda(self) -> float:
        return 1.0

    def _plan_cum(self, h: int) -> tuple[float, float]:
        if h <= 0:
            return 0.0, 0.0
        point = self.plan.trajectory[min(h, self.horizon) - 1]
        kpi = {"clicks": point.cum_clicks, "conversions": point.cum_conversions, "reach": point.cum_reach}[self.kpi]
        return point.cum_spend_rub, kpi

    def _tracking_errors(self, h: int) -> tuple[float, float]:
        plan_spend, plan_kpi = self._plan_cum(h)
        err_spend = (plan_spend - self.fact_cum_spend) / plan_spend if plan_spend > 0 else 0.0
        err_kpi = (plan_kpi - self.fact_cum_kpi) / plan_kpi if plan_kpi > 0 else 0.0
        return err_spend, err_kpi

    def _status(self, err_spend: float, err_kpi: float) -> TrackingStatus:
        if any(est.shock_active for est in self.estimates.values()):
            return TrackingStatus.FIRE
        band = max(self.plan.corridor_rel, DEAD_ZONE)
        if abs(err_spend) > band or abs(err_kpi) > band:
            return TrackingStatus.WATCH
        return TrackingStatus.OK

    def _channel_status(self, cid: str, cap: float) -> ChannelStatus:
        est = self.estimates[cid]
        if est.hours_without_delivery >= PAUSE_AFTER_HOURS and self.last_caps.get(cid, 0) > 0:
            return ChannelStatus.PAUSED
        if cap <= 0 and self.plan_budget[cid] > 0:
            return ChannelStatus.FROZEN_CAPACITY
        return ChannelStatus.ACTIVE

    def _reason(self, cid: str, cap: float, h: int) -> str:
        planned = self.plan_caps[h][cid] if h < self.horizon else 0.0
        if planned <= 0 and cap <= 0:
            return "по плану канал в этот час не работает"
        ratio = cap / planned if planned > 0 else 0.0
        return f"лимит {cap:,.0f} ₽ против плановых {planned:,.0f} ₽ (×{ratio:.2f})"

    def _plan_daily(self, cid: str) -> float:
        return self.plan_budget[cid] / max(self.horizon / 24, 1.0)

    def _rate_scale(self, cid: str, hour_index: int) -> float:
        """Во сколько раз отдача на рубль по кривой на намеченном дневном уровне отличается от плановой.

        Перерешение меняет расход канала, а цена показа по ландшафту зависит от объёма:
        без этой поправки детектор принял бы дешевеющий после урезания канал за «рост».
        """
        curve = self.curves[cid]
        share = curve.hourly_share(max(hour_index, 0))
        cap = self.last_caps.get(cid, 0.0)
        plan_daily = self._plan_daily(cid)
        intended = cap / share if share > 0 and cap > 0 else plan_daily

        def per_rub(daily: float) -> float:
            spend = curve.effective_spend(daily)
            return curve.impressions_at(daily) / spend if spend > 0 else 0.0

        base = per_rub(plan_daily)
        return per_rub(intended) / base if base > 0 else 1.0

    def _shadow_price(self, remaining_budget: float) -> float | None:
        plan_spend, plan_kpi = self._plan_cum(self.horizon)
        remaining_kpi = max(plan_kpi - self.fact_cum_kpi, 0.0)
        return remaining_budget / remaining_kpi if remaining_kpi > 0 else None

    def _profile_tail(self, cid: str, h: int) -> tuple[float, float]:
        """Доля планового расхода канала в час h и сумма долей с h до конца."""
        caps = [self.plan_caps[t][cid] for t in range(h, self.horizon)]
        total = sum(caps)
        return (caps[0] if caps else 0.0), total


def _clamp_total(caps: dict[str, float], remaining_budget: float) -> dict[str, float]:
    total = sum(caps.values())
    if total <= remaining_budget or total <= 0:
        return {cid: max(v, 0.0) for cid, v in caps.items()}
    scale = remaining_budget / total
    return {cid: max(v * scale, 0.0) for cid, v in caps.items()}


# ---------------------------------------------------------------- baselines


@dataclass
class StaticExecutor(BaseExecutor):
    name: str = "static"

    def _caps(self, h: int, remaining_budget: float) -> dict[str, float]:
        return dict(self.plan_caps[h])


@dataclass
class ProportionalExecutor(BaseExecutor):
    """Остаток бюджета канала по оставшемуся плановому профилю (Turn, формула 7)."""

    name: str = "proportional_pacing"

    def _caps(self, h: int, remaining_budget: float) -> dict[str, float]:
        caps = {}
        for cid in self.channel_ids:
            share, tail = self._profile_tail(cid, h)
            left = max(self.plan_budget[cid] - self.fact_cum_by_channel[cid], 0.0)
            caps[cid] = left * share / tail if tail > 0 else 0.0
        return caps


@dataclass
class PidExecutor(BaseExecutor):
    """PID по ошибке накопительного расхода канала, множитель к плановому лимиту."""

    name: str = "pid"
    integral: dict[str, float] = field(default_factory=dict)
    prev_error: dict[str, float] = field(default_factory=dict)

    def _caps(self, h: int, remaining_budget: float) -> dict[str, float]:
        caps = {}
        for cid in self.channel_ids:
            plan_cum = self.plan.trajectory[h - 1].by_channel_cum_spend_rub[cid] if h > 0 else 0.0
            error = (plan_cum - self.fact_cum_by_channel[cid]) / plan_cum if plan_cum > 0 else 0.0
            self.integral[cid] = self.integral.get(cid, 0.0) + error
            derivative = error - self.prev_error.get(cid, 0.0)
            self.prev_error[cid] = error
            multiplier = 1 + PID_KP * error + PID_KI * self.integral[cid] + PID_KD * derivative
            multiplier = min(max(multiplier, LAMBDA_MIN), LAMBDA_MAX)
            caps[cid] = self.plan_caps[h][cid] * multiplier
        return caps


# ----------------------------------------------------------------- adaptive


@dataclass
class AdaptiveExecutor(BaseExecutor):
    """Два контура: перерешение распределения остатка и удержание темпа."""

    name: str = "adaptive"
    auto_apply_above_limit: bool = True
    approved_hours: set[int] = field(default_factory=set)  # часы, в которые человек одобрил ход выше лимита
    rejected_hours: set[int] = field(default_factory=set)  # часы, чьи ходы человек отклонил или откатил
    frozen_donors: set[str] = field(default_factory=set)  # каналы, из которых нельзя забирать, пока их карточка ждёт или отклонена
    share_anchor: dict[str, float] = field(default_factory=dict)  # доли каналов на начало текущих суток
    hold_cards: set[str] = field(default_factory=set)  # каналы, по которым карточка «держим» уже выдана в этом сломе
    uncarded_rub: dict[str, float] = field(default_factory=dict)  # переносы по факту, ещё не показанные карточкой, по донорам
    last_taker_hour: dict[str, int] = field(default_factory=dict)  # когда канал последний раз получал деньги: гистерезис против пинг-понга
    last_card_hour: dict[str, int] = field(default_factory=dict)
    anchor_hour: int = -HOURS_IN_WEEK
    hold_plan: bool = True  # False = выжимать максимум KPI, резерв не используется
    lam: float = 1.0  # множитель темпа по расходу
    reserve_rub: float = 0.0  # бюджет, который решено не тратить: план по KPI и так выполняется
    target_budget: dict[str, float] = field(default_factory=dict)  # бюджет канала на всю кампанию после перерешений
    last_replan_hour: int = -REPLAN_EVERY_HOURS
    pending_replan: bool = False
    unavailable: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.target_budget = dict(self.plan_budget)

    def _after_observe(self, fired: list[str]) -> None:
        if fired:
            self.pending_replan = True
            # сломанный канал больше не защищён решением человека: причина хода изменилась
            self.frozen_donors.difference_update(fired)
        for cid in self.channel_ids:
            est = self.estimates[cid]
            if est.hours_without_delivery >= PAUSE_AFTER_HOURS and self.last_caps.get(cid, 0.0) > 0:
                if cid not in self.unavailable:
                    self.unavailable.add(cid)
                    self.pending_replan = True
            elif cid in self.unavailable and est.hours_without_delivery == 0:
                self.unavailable.discard(cid)
                self.events.append(f"час {self.hour}: канал {cid} снова отдаёт показы")
                self.pending_replan = True

    def _lambda(self) -> float:
        return self.lam

    def _caps(self, h: int, remaining_budget: float) -> dict[str, float]:
        if self.pending_replan or h - self.last_replan_hour >= REPLAN_EVERY_HOURS:
            self._update_lambda(h)  # темп корректируется блоками, а не каждый час: меньше дёрганья
            self._replan(h, remaining_budget)
        caps = {}
        for cid in self.channel_ids:
            share, tail = self._profile_tail(cid, h)
            left = max(self.target_budget[cid] - self.fact_cum_by_channel[cid], 0.0)
            base = left * share / tail if tail > 0 else 0.0
            if cid in self.unavailable:
                base = self.plan_caps[h][cid] * PROBE_SHARE  # щупаем канал, не тратим на него всерьёз
            caps[cid] = base * self.lam
        return caps

    # ------------------------------------------------ внутренний контур
    @staticmethod
    def _band_step(error: float) -> float:
        step = 0.0
        for threshold, size in ERROR_BANDS:
            if abs(error) >= threshold:
                step = size
        return step

    def _update_lambda(self, h: int) -> None:
        if h == 0:
            return
        err_spend, _ = self._tracking_errors(h)
        if abs(err_spend) >= DEAD_ZONE:
            step = self._band_step(err_spend)
            self.lam = min(max(self.lam * (1 + step * math.copysign(1.0, err_spend)), LAMBDA_MIN), LAMBDA_MAX)
        if self.reserve_rub > 0:
            # мы сознательно тратим меньше плана: не разгоняемся, чтобы «догнать» его
            self.lam = min(self.lam, 1.0)

    # -------------------------------------------------- внешний контур
    def _adjusted_curves(self, h: int) -> dict[str, ResponseCurve]:
        """Кривые каталога, поправленные фактом: цена, CTR и CVR по оценкам."""
        adjusted = {}
        for cid in self.channel_ids:
            curve = self.curves[cid]
            est = self.estimates[cid]
            price_ratio = 1.0
            if est.observed_ecpm and est.cum_impressions > 0:
                # ожидаемая цена берётся из кривой на фактическом дневном уровне расхода, а не из
                # середины каталога: кривая уже учитывает, что выкуп дорожает с объёмом
                days_left = max((self.horizon - h) / 24, 1.0)
                daily_level = max(self.target_budget[cid] - self.fact_cum_by_channel[cid], 0.0) / days_left or self._plan_daily(cid)
                imps = curve.impressions_at(daily_level)
                expected_ecpm = curve.effective_spend(daily_level) / imps * 1000 if imps > 0 else self.catalog.by_id(cid).ecpm_mid
                price_ratio = expected_ecpm / est.observed_ecpm  # дороже = меньше показов на рубль
                price_ratio = min(max(price_ratio, 0.3), 3.0)
            if cid in self.unavailable:
                price_ratio = 0.0
            points = [
                CurvePoint(p.daily_spend, p.impressions * price_ratio, p.clicks, p.conversions, p.reach)
                for p in curve.points
            ]
            adjusted[cid] = replace(
                curve,
                points=points,
                ctr=min(est.ctr.value, curve.ctr) if est.suspicious else est.ctr.value,
                cvr=est.cvr.value,
                max_daily_spend=curve.max_daily_spend if price_ratio > 0 else 0.0,
                max_daily_impressions=curve.max_daily_impressions * price_ratio,
            )
        return adjusted

    def _replan(self, h: int, remaining_budget: float) -> None:
        self.last_replan_hour = h
        self.pending_replan = False
        days_left = max(math.ceil((self.horizon - h) / 24), 1)
        if remaining_budget <= 0 or days_left <= 0:
            return
        curves = self._adjusted_curves(h)
        pools = {}
        for cid in self.channel_ids:
            total_pool = self.catalog.by_id(cid).capacity_mid * campaign_audience_multiplier()
            pools[cid] = max(total_pool - self.estimates[cid].cum_reach, total_pool * 0.1)
        models = build_models(curves, days_left, pools, fatigue_delta(), grid_size=REPLAN_GRID_SIZE)
        old_left = {cid: max(self.target_budget[cid] - self.fact_cum_by_channel[cid], 0.0) for cid in self.channel_ids}
        if self.hold_plan and h >= RESERVE_WARMUP_SHARE * self.horizon:
            wanted_reserve = remaining_budget - self._budget_to_use(models, remaining_budget)
            step = RESERVE_STEP_SHARE * remaining_budget
            self.reserve_rub = min(max(wanted_reserve, self.reserve_rub - step), self.reserve_rub + step, remaining_budget)
        else:
            self.reserve_rub = 0.0
        budget_to_use = remaining_budget - self.reserve_rub
        # гистерезис: канал, получивший деньги в последние сутки, не отдаёт их обратно по шуму
        # оценок; только слом (детектор, пауза) снимает защиту
        recent_takers = {
            cid for cid, t in self.last_taker_hour.items()
            if h - t < CARD_MIN_INTERVAL_HOURS and not self.estimates[cid].shock_active and cid not in self.unavailable
        }
        locked = {cid: min(old_left[cid], budget_to_use) for cid in self.frozen_donors | recent_takers if cid in old_left}
        if sum(locked.values()) > budget_to_use:
            scale = budget_to_use / sum(locked.values())
            locked = {cid: v * scale for cid, v in locked.items()}
        result = greedy_allocate(
            models, budget_to_use, self.kpi, locked=locked, max_cost_per_kpi=self.plan.brief.max_cpa_rub, steps=REPLAN_STEPS
        )
        new_left = dict(result.budgets)

        # ограничитель: доля канала меняется не более чем на ±30 % за сутки, а не за каждое
        # перерешение, иначе четыре перерешения в день дают четырёхкратный сдвиг по шуму первых часов
        total_old = sum(old_left.values()) or remaining_budget
        if not self.share_anchor or h - self.anchor_hour >= 24:
            self.share_anchor = {cid: (old_left[cid] / total_old if total_old > 0 else 0.0) for cid in self.channel_ids}
            self.anchor_hour = h
        for cid in self.channel_ids:
            if cid in self.unavailable:
                new_left[cid] = 0.0
                continue
            if cid in locked:
                continue
            old_share = self.share_anchor.get(cid, old_left[cid] / total_old if total_old > 0 else 0.0)
            new_share = new_left[cid] / remaining_budget if remaining_budget > 0 else 0.0
            lo, hi = old_share * (1 - SHARE_RATE_LIMIT), old_share * (1 + SHARE_RATE_LIMIT) + 0.01
            if self.estimates[cid].shock_active:
                lo = 0.0  # сломанный канал урезается сразу: плавность нужна для шума, а не для слома
            new_left[cid] = min(max(new_share, lo), hi) * remaining_budget
        # остаток после ограничений раздаётся только живым и не защищённым каналам:
        # сломанный канал и канал с ожидающей карточкой не должны получать его пропорционально
        adjustable = [
            cid for cid in self.channel_ids
            if cid not in self.unavailable and cid not in locked and not self.estimates[cid].shock_active
        ] or [cid for cid in self.channel_ids if cid not in self.unavailable]
        fixed = sum(v for cid, v in new_left.items() if cid not in adjustable)
        adjustable_sum = sum(new_left[cid] for cid in adjustable)
        scale = (budget_to_use - fixed) / adjustable_sum if adjustable_sum > 0 else 1.0
        new_left = {cid: (v * scale if cid in adjustable else v) for cid, v in new_left.items()}

        self._emit_proposals(h, old_left, new_left, models, remaining_budget)
        for cid in self.channel_ids:
            if new_left[cid] - old_left[cid] > PROPOSAL_MIN_SHARE * remaining_budget:
                self.last_taker_hour[cid] = h
            self.target_budget[cid] = self.fact_cum_by_channel[cid] + new_left[cid]

    def _budget_to_use(self, models, remaining_budget: float) -> float:
        """Сколько из остатка тратить: весь остаток, если план по KPI не перевыполняется, иначе меньше.

        Кейс просит удерживать факт на траектории плана, а не максимизировать KPI
        любой ценой, и меряет отклонение по расходу и по KPI с равным весом.
        Опережение оценивается по факту, а не по прогнозу модели: r = факт / план
        накопительно к текущему часу. Если остаток тратить по плану, KPI к концу
        превысит план примерно в r раз; доля резерва s выбирается так, чтобы
        относительный недорасход равнялся относительному перевыполнению:
        r(1 − s) − 1 = s, откуда s = (r − 1) / (r + 1). Сумма двух отклонений
        при этом минимальна.
        """
        _, plan_total = self._plan_cum(self.horizon)
        _, plan_to_date = self._plan_cum(self.hour)
        if plan_total - self.fact_cum_kpi <= 0:
            return 0.0
        if plan_to_date <= 0 or self.fact_cum_kpi <= 0:
            return remaining_budget
        ratio = self.fact_cum_kpi / plan_to_date
        if ratio <= 1 + DEAD_ZONE:
            return remaining_budget
        share = (ratio - 1) / (ratio + 1)
        return remaining_budget * (1 - share)

    def _emit_hold_cards(self, h: int, deltas: dict, models, old_left: dict, remaining_budget: float) -> None:
        """Слом есть, а переноса нет: человек всё равно должен увидеть карточку с объяснением.

        Так бывает, когда остальные каналы у потолка ёмкости и деньги некуда деть,
        или когда рост CTR не засчитан до подтверждения конверсиями. Карточка с нулевой
        суммой честно говорит: «держим, пересматриваем каждые 6 часов».
        """
        for cid in self.channel_ids:
            est = self.estimates[cid]
            if not est.shock_active:
                self.hold_cards.discard(cid)
                continue
            if cid in self.hold_cards or deltas.get(cid, 0.0) < 0 or cid in self.unavailable:
                continue
            self.hold_cards.add(cid)
            saturated = [c for c in self.channel_ids if c != cid and old_left.get(c, 0.0) >= models[c].max_budget * 0.97]
            if est.last_event == "rise":
                cause, kind = f"клики {cid} выросли при прежней цене и без роста конверсий: похоже на фрод, детектор в час {self.detection_hours.get(cid, h)}", "rise"
                decision = "деньги в канал не добавляем, пока рост не подтвердится конверсиями"
            else:
                cause, kind = f"отдача {cid} на рубль упала, детектор сработал в час {self.detection_hours.get(cid, h)}", "drop"
                decision = (
                    f"перенос невыгоден: {', '.join(saturated)} у потолка ёмкости, деньгам некуда уйти"
                    if saturated else "перенос пока невыгоден: остальные каналы дают не больше на рубль"
                )
            self.proposals.append(
                Proposal(
                    hour=h, from_channel=cid, to_channel=cid, amount_rub=0.0, cause=cause,
                    cost_of_decision=f"держим бюджет {cid}: {decision}; пересматриваем каждые {REPLAN_EVERY_HOURS} часов",
                    cost_of_inaction="бездействие и есть решение: цена та же",
                    cpa_delta_pct=0.0, inaction_kpi_shortfall_pct=0.0, inaction_kpi_shortfall_abs=0.0,
                    cause_kind=kind, applied_by="system",
                )
            )
            self.events.append(f"час {h}: {cause}; {decision}")

    def _emit_proposals(self, h: int, old_left: dict, new_left: dict, models, remaining_budget: float) -> None:
        deltas = {cid: new_left[cid] - old_left[cid] for cid in self.channel_ids}
        donors = sorted((cid for cid in deltas if deltas[cid] < 0), key=lambda c: deltas[c])
        takers = sorted((cid for cid in deltas if deltas[cid] > 0), key=lambda c: -deltas[c])
        self._emit_hold_cards(h, deltas, models, old_left, remaining_budget)
        if not donors or not takers:
            return
        # сломанный канал идёт первым и без порога по сумме: человек должен увидеть карточку в час слома
        broken = [cid for cid in donors if self.estimates[cid].shock_active or cid in self.unavailable]
        if broken:
            donors = broken + [cid for cid in donors if cid not in broken]
        kpi_old = sum(models[cid].value(min(old_left[cid], models[cid].max_budget), self.kpi) for cid in self.channel_ids)
        kpi_new = sum(models[cid].value(min(new_left[cid], models[cid].max_budget), self.kpi) for cid in self.channel_ids)
        cpa_old = remaining_budget / kpi_old if kpi_old > 0 else None
        cpa_new = remaining_budget / kpi_new if kpi_new > 0 else None
        cpa_delta = ((cpa_new - cpa_old) / cpa_old * 100) if cpa_old and cpa_new else None
        _, plan_kpi_total = self._plan_cum(self.horizon)
        plan_left = max(plan_kpi_total - self.fact_cum_kpi, 0.0)
        # цена бездействия: недобор к концу кампании в долях всего плана, а не остатка
        shortfall_abs = max(plan_left - kpi_old, 0.0)
        shortfall = (shortfall_abs / plan_kpi_total * 100) if plan_kpi_total > 0 else None
        limit = self.plan.brief.automation_limit_rub

        for donor in donors[:1]:  # одна карточка на перерешение: сломанный канал или самый крупный перенос
            amount = -deltas[donor]
            forced = donor in broken
            if amount < PROPOSAL_MIN_SHARE * remaining_budget and not forced:
                continue
            taker = takers[0]
            est = self.estimates[donor]
            if donor in self.unavailable:
                cause, kind = f"{donor} не отдаёт показы", "pause"
            elif est.shock_active and est.last_event == "rise":
                cause, kind = f"клики {donor} выросли при прежней цене и без роста конверсий: похоже на фрод, детектор в час {self.detection_hours.get(donor, h)}", "rise"
            elif est.shock_active:
                cause, kind = f"отдача {donor} на рубль упала, детектор сработал в час {self.detection_hours.get(donor, h)}", "drop"
            else:
                cause, kind = f"по факту {taker} даёт больше KPI на рубль, чем {donor}", "fact"
            applied_by = "system"
            if h in self.rejected_hours:
                applied_by = "rejected"
            elif limit is not None and amount > limit:
                self.human_requests += 1
                if self.auto_apply_above_limit:
                    applied_by = "system"
                elif h in self.approved_hours:
                    applied_by = "human"
                else:
                    applied_by = "pending"
            if applied_by == "system" and not forced:
                # перенос по факту применяется сразу, а карточка о нём копится: не чаще раза в
                # сутки на канал и не мельче порога, иначе человек читает один перенос по частям
                self.uncarded_rub[donor] = self.uncarded_rub.get(donor, 0.0) + amount
                if h - self.last_card_hour.get(donor, -CARD_MIN_INTERVAL_HOURS) < CARD_MIN_INTERVAL_HOURS or self.uncarded_rub[donor] < PROPOSAL_MIN_SHARE * remaining_budget:
                    continue
                amount = self.uncarded_rub[donor]
            self.uncarded_rub[donor] = 0.0
            self.last_card_hour[donor] = h
            self.proposals.append(
                Proposal(
                    hour=h,
                    from_channel=donor,
                    to_channel=taker,
                    amount_rub=float(amount),
                    cause=cause,
                    cost_of_decision=(f"CPA на остатке изменится на {cpa_delta:+.1f} %" if cpa_delta is not None else "CPA не оценён"),
                    cost_of_inaction=(
                        f"без хода недоберём {shortfall:.0f} % {self.kpi} к концу"
                        if shortfall is not None and shortfall > 0
                        else "без хода план выполняется; ход экономит бюджет"
                    ),
                    cpa_delta_pct=cpa_delta,
                    inaction_kpi_shortfall_pct=shortfall,
                    inaction_kpi_shortfall_abs=shortfall_abs,
                    cause_kind=kind,
                    applied_by=applied_by,
                )
            )
            self.events.append(f"час {h}: предложение перенести {amount:,.0f} ₽ из {donor} в {taker}: {cause}")
            self.hold_cards.add(donor)  # по этому слому карточка уже есть, «держим» не нужна
            if applied_by in ("pending", "rejected"):
                # человек не одобрил: ход не применяем, деньги остаются у донора, и донор заморожен,
                # чтобы автоматика не добрала тот же перенос частями ниже лимита
                new_left[taker] -= amount
                new_left[donor] += amount
                self.frozen_donors.add(donor)


POLICIES = {
    "static": StaticExecutor,
    "proportional_pacing": ProportionalExecutor,
    "pid": PidExecutor,
    "adaptive": AdaptiveExecutor,
}


def make_executor(name: str, plan: MediaPlan, catalog: PublicCatalog, curves: dict[str, ResponseCurve], total_budget: float, **kwargs) -> BaseExecutor:
    try:
        cls = POLICIES[name]
    except KeyError as exc:
        raise KeyError(f"неизвестная стратегия {name!r}; есть: {sorted(POLICIES)}") from exc
    return cls(plan=plan, catalog=catalog, curves=curves, total_budget=total_budget, **kwargs)


def kpi_of_observation(obs: Observation, kpi: str) -> float:
    return float(sum(_kpi_of(ch, kpi) for ch in obs.by_channel.values()))


def spend_by_channel(obs: Observation) -> dict[str, float]:
    return {cid: ch.spend for cid, ch in obs.by_channel.items()}


def as_array(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)
