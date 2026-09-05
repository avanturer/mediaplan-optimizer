"""Планировщик: бриф → медиаплан или диагноз недостижимости.

Тип A: жадное распределение заданного бюджета. Тип B: бисекция по бюджету
поверх того же алгоритма (KPI(B) монотонна и вогнута), как режимы
``fixed_budget`` и ``target_*`` у Meridian и ``max_response`` /
``target_efficiency`` у Robyn. Отказ называет связывающее ограничение и
приносит три готовых альтернативы с посчитанным результатом (сценарий С2).
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from brain.assumptions import campaign_audience_multiplier, fatigue_delta
from brain.config import CORRIDOR_SIGMA_DIVISOR
from brain.curves import ResponseCurve
from brain.ml import MLBundle, ReachModel
from brain.planner.allocator import AllocationResult, ChannelModel, allocate, build_models
from contracts import (
    BindingConstraint,
    Brief,
    BriefSuggestion,
    CalendarCell,
    ChannelAllocation,
    Forecast,
    Infeasibility,
    MediaPlan,
    PublicCatalog,
    TrajectoryPoint,
)


@dataclass
class PlanningContext:
    catalog: PublicCatalog
    curves: dict[str, ResponseCurve]
    channel_ids: list[str]
    days: int
    reach_model: ReachModel | None = None
    ml_model_id: str | None = None

    def models(self, days: int | None = None) -> dict[str, ChannelModel]:
        d = days or self.days
        pools = {
            cid: self.catalog.by_id(cid).capacity_mid * campaign_audience_multiplier()
            for cid in self.channel_ids
        }
        return build_models(
            {cid: self.curves[cid] for cid in self.channel_ids}, d, pools, fatigue_delta()
        )


def plan(brief: Brief, catalog: PublicCatalog, curves: dict[str, ResponseCurve], ml_bundle: MLBundle | None = None) -> MediaPlan:
    if brief.targeting != catalog.targeting:
        raise ValueError("таргетинг брифа и каталога должен совпадать; соберите ретро-историю выбранного сегмента")
    if brief.ml.enabled:
        if ml_bundle is None or ml_bundle.catalog_id != catalog.catalog_id:
            raise ValueError("для ML нужен обученный артефакт выбранного сегмента")
        if brief.ml.response_curves:
            curves = ml_bundle.curves
    missing = [cid for cid in brief.channel_ids if cid not in curves]
    if missing:
        raise ValueError(f"нет кривых для каналов {missing}")
    reach_model = ml_bundle.reach if brief.ml.reach_correction and ml_bundle else None
    model_id = ml_bundle.model_id if brief.ml.enabled and ml_bundle else None
    ctx = PlanningContext(catalog, curves, list(brief.channel_ids), brief.horizon_days, reach_model, model_id)
    models = ctx.models()
    kpi = brief.kpi_name

    if brief.is_budget_constrained:
        assert brief.budget_rub is not None
        result = allocate(models, brief.budget_rub, kpi, brief.locked, brief.max_cpa_rub, reach_model=reach_model)
        return _assemble(brief, catalog, ctx, models, result, kpi)

    assert brief.target_value is not None
    diagnosis = _diagnose(brief, catalog, ctx, models, kpi)
    if diagnosis is not None:
        return MediaPlan(
            plan_id=_plan_id(brief, catalog) + (f"-{model_id[:8]}" if model_id else ""),
            ml_model_id=model_id,
            catalog_id=catalog.catalog_id,
            brief=brief,
            kpi_name=kpi,
            infeasibility=diagnosis,
        )
    budget = _min_budget_for(models, brief.target_value, kpi, brief.locked, brief.max_cpa_rub, reach_model)
    result = allocate(models, budget, kpi, brief.locked, brief.max_cpa_rub, reach_model=reach_model)
    return _assemble(brief, catalog, ctx, models, result, kpi)


# ----------------------------------------------------------------- тип B


def _total_kpi(models: dict[str, ChannelModel], budget: float, kpi: str, locked, max_cpa, reach_model=None) -> float:
    result = allocate(models, budget, kpi, locked, max_cpa, steps=300, reach_model=reach_model)
    if kpi == "reach" and reach_model is not None:
        return reach_model.predict({cid: models[cid].value(b, kpi) for cid, b in result.budgets.items()})
    return sum(models[cid].value(b, kpi) for cid, b in result.budgets.items())


def _max_budget(models: dict[str, ChannelModel]) -> float:
    return sum(m.max_budget for m in models.values())


def _min_budget_for(models, target: float, kpi: str, locked, max_cpa, reach_model=None) -> float:
    lo, hi = 0.0, _max_budget(models)
    for _ in range(40):
        mid = (lo + hi) / 2
        if _total_kpi(models, mid, kpi, locked, max_cpa, reach_model) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _diagnose(brief: Brief, catalog: PublicCatalog, ctx: PlanningContext, models, kpi: str) -> Infeasibility | None:
    target = brief.target_value
    assert target is not None
    max_kpi = _total_kpi(models, _max_budget(models), kpi, brief.locked, brief.max_cpa_rub, ctx.reach_model)
    if max_kpi >= target:
        return None

    suggestions: list[BriefSuggestion] = []
    # 1. увеличить срок: минимальный горизонт, при котором цель достижима
    min_days = None
    for days in range(brief.horizon_days + 1, 91):
        m = ctx.models(days)
        if _total_kpi(m, _max_budget(m), kpi, brief.locked, brief.max_cpa_rub, ctx.reach_model) >= target:
            min_days = days
            budget = _min_budget_for(m, target, kpi, brief.locked, brief.max_cpa_rub, ctx.reach_model)
            suggestions.append(
                BriefSuggestion(
                    description=f"Увеличить срок до {days} дней",
                    changed_field="horizon_days",
                    suggested_value=days,
                    expected_kpi=float(target),
                    expected_budget_rub=float(budget),
                )
            )
            break
    # 2. снизить цель до достижимого максимума с запасом 5 %
    reachable = max_kpi * 0.95
    budget_for_reachable = _min_budget_for(models, reachable, kpi, brief.locked, brief.max_cpa_rub, ctx.reach_model)
    suggestions.append(
        BriefSuggestion(
            description=f"Снизить цель до {reachable:,.0f} {kpi} в срок",
            changed_field="target_value",
            suggested_value=float(round(reachable)),
            expected_kpi=float(reachable),
            expected_budget_rub=float(budget_for_reachable),
        )
    )
    # 3. добавить каналы, которых нет в пресете
    extra = [cid for cid in catalog.channel_ids if cid not in brief.channel_ids and cid in ctx.curves]
    if extra:
        wide = PlanningContext(catalog, ctx.curves, list(brief.channel_ids) + extra, brief.horizon_days, ctx.reach_model, ctx.ml_model_id)
        wide_models = wide.models()
        wide_max = _total_kpi(wide_models, _max_budget(wide_models), kpi, brief.locked, brief.max_cpa_rub, ctx.reach_model)
        if wide_max >= target:
            budget = _min_budget_for(wide_models, target, kpi, brief.locked, brief.max_cpa_rub, ctx.reach_model)
            suggestions.append(
                BriefSuggestion(
                    description=f"Добавить каналы: {', '.join(extra)}",
                    changed_field="channel_ids",
                    suggested_value=list(brief.channel_ids) + extra,
                    expected_kpi=float(target),
                    expected_budget_rub=float(budget),
                )
            )
            constraint = BindingConstraint.CHANNEL_SET
            explanation = (
                f"При выбранных каналах максимум за {brief.horizon_days} дней "
                f"{max_kpi:,.0f} {kpi}; с добавлением {', '.join(extra)} цель достижима."
            )
            return Infeasibility(
                binding_constraint=constraint, explanation=explanation, max_achievable=max_kpi, suggestions=suggestions
            )

    if min_days is not None:
        constraint = BindingConstraint.HORIZON
        explanation = (
            f"Ёмкости каналов хватает, но не за {brief.horizon_days} дней: потолок "
            f"{max_kpi:,.0f} {kpi}; цель достижима минимум за {min_days} дней."
        )
    else:
        constraint = BindingConstraint.CAPACITY
        explanation = (
            f"Суммарная ёмкость выбранных каналов даёт не более {max_kpi:,.0f} {kpi} "
            f"даже при максимальном выкупе; цель {target:,.0f} недостижима."
        )
    return Infeasibility(
        binding_constraint=constraint, explanation=explanation, max_achievable=max_kpi, suggestions=suggestions
    )


# --------------------------------------------------------------- сборка


def _assemble(brief: Brief, catalog: PublicCatalog, ctx: PlanningContext, models, result: AllocationResult, kpi: str) -> MediaPlan:
    days = brief.horizon_days
    hours = days * 24
    allocations: list[ChannelAllocation] = []
    calendar: list[CalendarCell] = []
    hourly_caps: list[dict[str, float]] = [{} for _ in range(hours)]
    per_channel_cum_spend = {cid: np.zeros(hours) for cid in ctx.channel_ids}
    per_channel_cum_reach = {cid: np.zeros(hours) for cid in ctx.channel_ids}
    cum = {k: np.zeros(hours) for k in ("spend", "impressions", "clicks", "conversions", "reach")}

    budget_total = sum(result.budgets.values())
    corridor_rel = 0.0
    for cid in ctx.channel_ids:
        model = models[cid]
        channel = catalog.by_id(cid)
        b = result.budgets[cid]
        out = model.outcome(b)
        daily = b / days
        hourly_spend = np.array(
            [daily * ctx.curves[cid].hourly_share(h) for h in range(hours)]
        )  # профиль нормирован внутри суток, сумма по дню = daily
        cum_spend = np.cumsum(hourly_spend)
        per_channel_cum_spend[cid] = cum_spend
        cum["spend"] += cum_spend
        # KPI по часам: дневное накопление из модели (усталость по дням),
        # внутри суток пропорционально доле расхода часа. Пропорция «KPI ∝ расход»
        # на всю кампанию завышала бы хвост: свежая аудитория отвечает лучше.
        for key in ("impressions", "clicks", "conversions", "reach"):
            daily_cum = out.daily_cum[key]
            for day in range(days):
                prev = daily_cum[day - 1] if day > 0 else 0.0
                day_total = daily_cum[day] - prev
                block = hourly_spend[day * 24 : (day + 1) * 24]
                within = np.cumsum(block) / block.sum() if block.sum() > 0 else np.linspace(1 / 24, 1, 24)
                cum[key][day * 24 : (day + 1) * 24] += prev + day_total * within
                if key == "reach":
                    per_channel_cum_reach[cid][day * 24 : (day + 1) * 24] = prev + day_total * within
        for h in range(hours):
            hourly_caps[h][cid] = float(hourly_spend[h])
        for day in range(1, days + 1):
            calendar.append(CalendarCell(day=day, channel_id=cid, budget_rub=float(daily)))

        spend_eff = out.spend if out.spend > 0 else b
        marginal = result.marginal_cost_per_kpi.get(cid)
        daily_spend = out.daily_cum["spend"]
        daily_clicks = out.daily_cum["clicks"]
        clicks_per_rub = []
        for d in range(days):
            s_prev = daily_spend[d - 1] if d > 0 else 0.0
            c_prev = daily_clicks[d - 1] if d > 0 else 0.0
            ds, dc = daily_spend[d] - s_prev, daily_clicks[d] - c_prev
            clicks_per_rub.append(float(dc / ds) if ds > 0 else 0.0)
        allocations.append(
            ChannelAllocation(
                channel_id=cid,
                display_name=channel.display_name,
                budget_rub=float(b),
                impressions=float(out.impressions),
                unique_reach=float(out.reach),
                clicks=float(out.clicks),
                conversions=float(out.conversions),
                ctr=float(out.clicks / out.impressions) if out.impressions else 0.0,
                cvr=float(out.conversions / out.clicks) if out.clicks else 0.0,
                vtr=0.35 if channel.supports_video else None,
                cpm_rub=float(spend_eff / out.impressions * 1000) if out.impressions else 0.0,
                cpc_rub=float(spend_eff / out.clicks) if out.clicks else None,
                cpa_rub=float(spend_eff / out.conversions) if out.conversions else None,
                frequency=float(out.impressions / out.reach) if out.reach else 0.0,
                capacity_utilization=float(min(b / model.max_budget, 1.0)) if model.max_budget else 0.0,
                marginal_cost_per_1000_kpi_rub=float(marginal * 1000) if marginal else None,
                expected_daily_clicks_per_rub=clicks_per_rub,
                locked=cid in brief.locked,
            )
        )
        if budget_total > 0:
            # диапазон каталога трактуем как P10–P90; коридор ±1σ = полуширина / z(0.90)
            corridor_rel += ctx.curves[cid].uncertainty / CORRIDOR_SIGMA_DIVISOR * (b / budget_total)

    if ctx.reach_model is not None:
        cum["reach"] = np.array([ctx.reach_model.predict({cid: values[h] for cid, values in per_channel_cum_reach.items()}) for h in range(hours)])
    kpi_curve = cum[kpi]
    trajectory = [
        TrajectoryPoint(
            hour=h + 1,
            cum_spend_rub=float(cum["spend"][h]),
            cum_impressions=float(cum["impressions"][h]),
            cum_clicks=float(cum["clicks"][h]),
            cum_conversions=float(cum["conversions"][h]),
            cum_reach=float(cum["reach"][h]),
            band_low_spend_rub=float(cum["spend"][h] * (1 - corridor_rel)),
            band_high_spend_rub=float(cum["spend"][h] * (1 + corridor_rel)),
            band_low_kpi=float(kpi_curve[h] * (1 - corridor_rel)),
            band_high_kpi=float(kpi_curve[h] * (1 + corridor_rel)),
            by_channel_cum_spend_rub={cid: float(per_channel_cum_spend[cid][h]) for cid in ctx.channel_ids},
        )
        for h in range(hours)
    ]
    total_kpi = float(kpi_curve[-1])
    forecast = Forecast(
        kpi_name=kpi,
        p10=total_kpi * (1 - corridor_rel),
        p50=total_kpi,
        p90=total_kpi * (1 + corridor_rel),
        probability_of_target=(
            None if brief.target_value is None else _probability(total_kpi, corridor_rel, brief.target_value)
        ),
    )
    explanation = list(result.steps)
    explanation.append("ML: прогноз охвата скорректирован моделью, обученной на общих охватах ретро-кампаний."
                       if ctx.reach_model else "Прогноз охвата суммирует каналы и не учитывает межканальные пересечения; фактический охват кампании дедуплицируется.")
    if brief.ml.response_curves:
        explanation.append("ML: распределение использует обученные кривые показов, кликов и конверсий по бюджету.")
    if result.unspent > 0:
        explanation.append(
            f"не распределено {result.unspent:,.0f} ₽: все каналы упёрлись в потолок или в лимит цены"
        )
    return MediaPlan(
        plan_id=_plan_id(brief, catalog) + (f"-{ctx.ml_model_id[:8]}" if ctx.ml_model_id else ""),
        ml_model_id=ctx.ml_model_id,
        catalog_id=catalog.catalog_id,
        brief=brief,
        kpi_name=kpi,
        allocations=allocations,
        calendar=calendar,
        trajectory=trajectory,
        hourly_caps=hourly_caps,
        forecast=forecast,
        corridor_rel=float(corridor_rel),
        explanation=explanation,
    )


def _probability(p50: float, rel: float, target: float) -> float:
    """Вероятность выполнить цель при нормальном разбросе с σ ≈ коридор / 1.28 (P10–P90)."""
    if p50 <= 0:
        return 0.0
    sigma = max(p50 * rel, 1e-9)  # rel уже равен ±1σ
    from statistics import NormalDist

    return float(1 - NormalDist(p50, sigma).cdf(target))


def _plan_id(brief: Brief, catalog: PublicCatalog) -> str:
    payload = json.dumps({"brief": brief.model_dump(mode="json"), "catalog": catalog.catalog_id}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
