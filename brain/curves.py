"""Кривые отклика «дневной бюджет → показы, клики, конверсии», построенные из ретро-наблюдений.

Планировщик не получает кривых от мира: он строит их сам из публичной
ретро-истории (contracts/retro.py), как Google Reach Planner строит прогноз
«по истории похожих кампаний». Семантика кривой, о которой команда
договорилась словами:

1. кривая описывает отклик за один день при частоте около 1 (каждый пробный
   эпизод начинается с чистого состояния);
2. с бюджетом масштабируется объём, ставки отклика (CTR, CVR) считаются
   постоянными; усталость от частоты накапливает планировщик по дням;
3. первая точка около нуля, последняя равна потолку: там, где расход
   перестаёт расти с лимитом, кривая заканчивается; за неё не экстраполируем.
"""

from dataclasses import dataclass, field

import numpy as np

from brain.config import PRIOR_CLICKS, PRIOR_IMPRESSIONS, SATURATION_TOLERANCE
from contracts.catalog import CatalogChannel, PublicCatalog
from contracts.retro import RetroHistory

HOURS_IN_WEEK = 168


@dataclass
class CurvePoint:
    daily_spend: float
    impressions: float
    clicks: float
    conversions: float
    reach: float


@dataclass
class ResponseCurve:
    """Кусочно-линейная кривая одного канала плюс оценки ставок и профиля."""

    channel_id: str
    points: list[CurvePoint]
    ctr: float
    cvr: float
    hourly_profile: np.ndarray  # 168 долей, внутри суток сумма 1
    reach_per_impression: float  # доля новых уникальных на показ при малой частоте
    max_daily_spend: float
    max_daily_impressions: float
    uncertainty: float = field(default=0.2)

    def impressions_at(self, daily_spend: float) -> float:
        spend = min(max(daily_spend, 0.0), self.max_daily_spend)
        xs = [p.daily_spend for p in self.points]
        ys = [p.impressions for p in self.points]
        return float(np.interp(spend, xs, ys))

    def effective_spend(self, daily_spend: float) -> float:
        """Деньги сверх потолка канала не тратятся: мир их просто не выкупит."""
        return min(max(daily_spend, 0.0), self.max_daily_spend)

    def hourly_share(self, hour: int) -> float:
        return float(self.hourly_profile[hour % HOURS_IN_WEEK])


def _daily_aggregates(history: RetroHistory, channel_id: str) -> list[CurvePoint]:
    """Одна точка на уровень лимита: средние дневные значения по всем ретро-кампаниям.

    Эпизоды с одинаковым дневным лимитом (разные прошлые кампании) усредняются.
    Если оставить их отдельными точками, разброс между кампаниями превращает
    кривую в лестницу с плоскими ступенями и вертикальными скачками, и жадный
    алгоритм, который опирается на вогнутость, строит вырожденный план.
    """
    groups: dict[float, list[dict[str, float]]] = {}
    for episode in history.episodes:
        days = max(episode.horizon_hours // 24, 1)
        cap_total = sum(a.spend_caps.get(channel_id, 0.0) for a in episode.actions)
        if cap_total <= 0:
            continue
        cap_per_day = round(cap_total / days, 2)
        groups.setdefault(cap_per_day, []).append(
            {
                "spend": sum(o.by_channel[channel_id].spend for o in episode.observations) / days,
                "impressions": sum(o.by_channel[channel_id].impressions for o in episode.observations) / days,
                "clicks": sum(o.by_channel[channel_id].clicks for o in episode.observations) / days,
                "conversions": sum(o.by_channel[channel_id].conversions for o in episode.observations) / days,
                "reach": sum(o.by_channel[channel_id].unique_reach for o in episode.observations) / days,
            }
        )
    points: list[CurvePoint] = []
    for cap_per_day, rows in groups.items():
        n = len(rows)
        point = CurvePoint(
            daily_spend=sum(r["spend"] for r in rows) / n,
            impressions=sum(r["impressions"] for r in rows) / n,
            clicks=sum(r["clicks"] for r in rows) / n,
            conversions=sum(r["conversions"] for r in rows) / n,
            reach=sum(r["reach"] for r in rows) / n,
        )
        point.__dict__["cap_per_day"] = cap_per_day  # связывал ли лимит: нужно для потолка
        points.append(point)
    return sorted(points, key=lambda p: p.daily_spend)


def _hourly_profile(history: RetroHistory, channel_id: str) -> np.ndarray:
    """Профиль по часам недели из наблюдаемых запросов, нормирован внутри суток."""
    sums = np.zeros(HOURS_IN_WEEK)
    counts = np.zeros(HOURS_IN_WEEK)
    for episode in history.episodes:
        for obs in episode.observations:
            hour = (obs.hour - 1) % HOURS_IN_WEEK
            sums[hour] += obs.by_channel[channel_id].requests
            counts[hour] += 1
    mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    profile = np.empty(HOURS_IN_WEEK)
    for day in range(7):
        block = mean[day * 24 : (day + 1) * 24]
        total = block.sum()
        profile[day * 24 : (day + 1) * 24] = block / total if total > 0 else np.full(24, 1 / 24)
    return profile


def _concave_hull(points: list[CurvePoint]) -> list[CurvePoint]:
    """Верхняя вогнутая огибающая: предельная отдача показов не растёт с расходом.

    Точка, после которой наклон увеличивается, это шум ретро-выборки, а не
    свойство рынка (закупка дорожает по мере выкупа, не дешевеет). Такие точки
    выбрасываются, пока наклоны не станут невозрастающими.
    """
    hull = list(points)
    changed = True
    while changed and len(hull) > 2:
        changed = False
        for i in range(1, len(hull) - 1):
            left = hull[i - 1]
            mid = hull[i]
            right = hull[i + 1]
            slope_in = (mid.impressions - left.impressions) / max(mid.daily_spend - left.daily_spend, 1e-9)
            slope_out = (right.impressions - mid.impressions) / max(right.daily_spend - mid.daily_spend, 1e-9)
            if slope_out > slope_in + 1e-9:
                del hull[i]
                changed = True
                break
    return hull


def build_curve(history: RetroHistory, channel: CatalogChannel) -> ResponseCurve:
    raw = _daily_aggregates(history, channel.channel_id)
    if not raw:
        raise ValueError(f"в ретро-истории нет данных по каналу {channel.channel_id}")

    # Потолок: первая точка, где расход заметно ниже лимита, значит лимит не связывал.
    saturated = [p for p in raw if p.daily_spend < SATURATION_TOLERANCE * p.__dict__["cap_per_day"]]
    if saturated:
        max_spend = max(p.daily_spend for p in saturated)
        max_imps = max(p.impressions for p in saturated)
    else:
        max_spend = raw[-1].daily_spend
        max_imps = raw[-1].impressions

    # Монотонная огибающая: показы не убывают с расходом (иначе оптимизатор поедет).
    points = [CurvePoint(0.0, 0.0, 0.0, 0.0, 0.0)]
    best_imps = 0.0
    for p in raw:
        if p.daily_spend <= points[-1].daily_spend + 1e-9:
            continue
        best_imps = max(best_imps, p.impressions)
        points.append(CurvePoint(p.daily_spend, best_imps, p.clicks, p.conversions, p.reach))
        if p.daily_spend >= max_spend - 1e-9:
            break

    points = _concave_hull(points)

    total_imps = sum(p.impressions for p in raw)
    total_clicks = sum(p.clicks for p in raw)
    total_conv = sum(p.conversions for p in raw)
    total_reach = sum(p.reach for p in raw)
    ctr = total_clicks / total_imps if total_imps else channel.ctr_mid
    cvr = total_conv / total_clicks if total_clicks else channel.cvr_mid
    # Байесовское сглаживание ставок приором из каталога: пробные эпизоды короткие,
    # и на редких конверсиях наивная оценка шумит (см. docs/research.md).
    ctr = (total_clicks + PRIOR_IMPRESSIONS * channel.ctr_mid) / (total_imps + PRIOR_IMPRESSIONS)
    cvr = (total_conv + PRIOR_CLICKS * channel.cvr_mid) / (total_clicks + PRIOR_CLICKS)

    return ResponseCurve(
        channel_id=channel.channel_id,
        points=points,
        ctr=float(min(max(ctr, 0.0), 1.0)),
        cvr=float(min(max(cvr, 0.0), 1.0)),
        hourly_profile=_hourly_profile(history, channel.channel_id),
        reach_per_impression=float(total_reach / total_imps) if total_imps else 0.5,
        max_daily_spend=float(max_spend),
        max_daily_impressions=float(max_imps),
        uncertainty=channel.relative_uncertainty,
    )


def build_curves(history: RetroHistory, catalog: PublicCatalog, channel_ids: list[str] | None = None) -> dict[str, ResponseCurve]:
    ids = channel_ids or catalog.channel_ids
    return {cid: build_curve(history, catalog.by_id(cid)) for cid in ids}
