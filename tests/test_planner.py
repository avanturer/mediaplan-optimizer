"""Планировщик: демо 1, демо 2, диагностика, фиксация канала, устойчивость к сетке."""

import time

import numpy as np

from brain.assumptions import fatigue_delta
from brain.planner import plan
from brain.planner.allocator import allocate
from brain.planner.planner import PlanningContext
from contracts import BindingConstraint, Brief, TargetKpi


def test_demo1_plan_is_complete_and_fast(catalog, curves, demo_brief):
    started = time.perf_counter()
    p = plan(demo_brief, catalog, curves)
    assert time.perf_counter() - started < 2.0
    assert p.is_feasible
    assert abs(p.total_budget_rub - 1_200_000) < 1_200_000 * 0.01
    assert len(p.allocations) == 8 and len(p.trajectory) == 21 * 24 and len(p.hourly_caps) == 21 * 24
    assert p.total_kpi > 0 and p.forecast is not None and p.forecast.p10 < p.forecast.p50 < p.forecast.p90
    for a in p.allocations:
        assert a.ctr > 0 and a.cvr > 0 and a.cpm_rub > 0
        assert a.cpa_rub is None or a.cpa_rub > 0
        assert 0 <= a.capacity_utilization <= 1
    assert abs(sum(sum(h.values()) for h in p.hourly_caps) - p.total_budget_rub) < 1.0
    assert p.explanation, "план должен объяснять порядок наливания"


def test_marginal_costs_are_equalised_across_active_channels(demo_plan):
    """Равенство предельных отдач: у каналов, не упёршихся в потолок, «цена следующей тысячи» близка."""
    active = [a for a in demo_plan.allocations if a.capacity_utilization < 0.8 and a.marginal_cost_per_1000_kpi_rub]
    costs = np.array([a.marginal_cost_per_1000_kpi_rub for a in active])
    assert len(active) >= 4
    assert costs.max() / costs.min() < 1.6, costs


def test_trajectory_is_cumulative_with_corridor(demo_plan):
    spend = [t.cum_spend_rub for t in demo_plan.trajectory]
    kpi = [t.cum_conversions for t in demo_plan.trajectory]
    assert all(b >= a for a, b in zip(spend, spend[1:], strict=False))
    assert all(b >= a for a, b in zip(kpi, kpi[1:], strict=False))
    last = demo_plan.trajectory[-1]
    assert last.band_low_spend_rub < last.cum_spend_rub < last.band_high_spend_rub
    assert 0 < demo_plan.corridor_rel < 0.5


def test_demo2_type_b_finds_sufficient_budget(catalog, curves):
    brief = Brief(target_kpi=TargetKpi.CLICKS, target_value=50_000, horizon_days=14, channel_ids=catalog.channel_ids)
    p = plan(brief, catalog, curves)
    assert p.is_feasible
    assert p.total_kpi >= 50_000 * 0.99
    assert p.forecast is not None and p.forecast.probability_of_target is not None


def test_type_b_diagnoses_binding_constraint(catalog, curves):
    """Узкий пресет: отказ с диагнозом и тремя ходами, каждый с посчитанным результатом."""
    brief = Brief(
        target_kpi=TargetKpi.CLICKS,
        target_value=50_000,
        horizon_days=14,
        channel_ids=["social_2", "social_3", "marketplace_1", "sms"],
    )
    p = plan(brief, catalog, curves)
    assert not p.is_feasible
    diag = p.infeasibility
    assert diag.binding_constraint in set(BindingConstraint)
    assert 0 < diag.max_achievable < 50_000
    kinds = {s.changed_field for s in diag.suggestions}
    assert {"horizon_days", "target_value", "channel_ids"} <= kinds
    for s in diag.suggestions:
        assert s.expected_budget_rub > 0 and s.expected_kpi > 0
    # цель, ниже потолка: считается
    ok = plan(brief.model_copy(update={"target_value": diag.max_achievable * 0.9}), catalog, curves)
    assert ok.is_feasible


def test_locked_channel_is_respected(catalog, curves, demo_brief):
    locked = demo_brief.model_copy(update={"locked": {"sms": 50_000.0}})
    p = plan(locked, catalog, curves)
    sms = next(a for a in p.allocations if a.channel_id == "sms")
    assert abs(sms.budget_rub - 50_000) < 1.0 and sms.locked
    assert abs(p.total_budget_rub - 1_200_000) < 1_200_000 * 0.01


def test_max_cpa_freezes_expensive_channels(catalog, curves, demo_brief):
    capped = demo_brief.model_copy(update={"max_cpa_rub": 500.0})
    p = plan(capped, catalog, curves)
    # Лимит задан на среднюю цену конверсии; при вогнутой кривой средняя цена ниже предельной,
    # а предельная цена следующей порции у остановленного канала по построению выше лимита.
    assert p.total_budget_rub / p.total_kpi <= 500.0
    for a in p.allocations:
        if a.budget_rub > 0 and a.cpa_rub:
            assert a.cpa_rub <= 500.0


def test_allocation_stable_under_finer_steps(catalog, curves, demo_brief):
    """Число порций и сетка модели это численное разрешение, не параметр результата."""
    ctx = PlanningContext(catalog, curves, list(catalog.channel_ids), 21)
    pools = {cid: catalog.by_id(cid).capacity_mid for cid in catalog.channel_ids}
    from brain.assumptions import campaign_audience_multiplier
    from brain.planner.allocator import build_models

    pools = {cid: v * campaign_audience_multiplier() for cid, v in pools.items()}
    coarse = allocate(build_models(ctx.curves, 21, pools, fatigue_delta(), grid_size=160), 1_200_000, "conversions", steps=1000)
    fine = allocate(build_models(ctx.curves, 21, pools, fatigue_delta(), grid_size=320), 1_200_000, "conversions", steps=2000)
    for cid in coarse.budgets:
        assert abs(coarse.budgets[cid] - fine.budgets[cid]) <= 0.02 * 1_200_000, cid
