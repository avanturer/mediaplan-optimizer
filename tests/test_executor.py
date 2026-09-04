"""Сервис исполнения: baseline'ы, детектор, эвакуация из сломанного канала, польза против static."""

import time

import numpy as np

from contracts import SeedBundle
from harness.compare import compare_strategies
from harness.runner import RunConfig, run_campaign

SEEDS = 6  # парных прогонов в тестах; полный стенд на 20–30 сидах запускает scripts/run_demos.py


def _cfg(strategy, scenario="stable", world_seed=1):
    return RunConfig(strategy=strategy, scenario_id=scenario, seeds=SeedBundle(world_seed=world_seed, noise_seed=10_000 + world_seed))


def test_baselines_complete_episode(demo_plan, catalog, curves):
    for strategy in ("static", "proportional_pacing", "pid", "adaptive"):
        summary = run_campaign(demo_plan, catalog, curves, _cfg(strategy))
        assert len(summary.hours) == 21 * 24
        assert summary.actual_spend <= demo_plan.total_budget_rub + 1e-6
        assert summary.final_deviation_spend <= 0.20 and summary.final_deviation_kpi <= 0.20


def test_full_run_is_fast(demo_plan, catalog, curves):
    started = time.perf_counter()
    run_campaign(demo_plan, catalog, curves, _cfg("adaptive"))
    assert time.perf_counter() - started < 5.0


def test_detector_silent_without_shock(demo_plan, catalog, curves):
    alarms = 0
    for ws in range(1, SEEDS + 1):
        alarms += len(run_campaign(demo_plan, catalog, curves, _cfg("static", "stable", ws)).detection_hours)
    assert alarms / SEEDS <= 0.5, f"ложных тревог на кампанию: {alarms / SEEDS}"


def test_detector_catches_ctr_drop(demo_plan, catalog, curves):
    """CTR −40 % в крупном канале с 240-го часа: детектор видит слом в течение двух суток в большинстве миров."""
    hits, delays = 0, []
    for ws in range(1, SEEDS + 1):
        summary = run_campaign(demo_plan, catalog, curves, _cfg("static", "ctr_drop", ws))
        hour = summary.detection_hours.get("marketplace_1")
        if hour is not None and hour >= 240:
            hits += 1
            delays.append(hour - 240)
    assert hits >= SEEDS * 0.7, f"ловим {hits} из {SEEDS}"
    assert max(delays) <= 72


def test_estimates_reset_after_shock(demo_plan, catalog, curves):
    from brain.executor import make_executor

    ex = make_executor("adaptive", demo_plan, catalog, curves, demo_plan.total_budget_rub)
    est = ex.estimates["marketplace_1"]
    est.ctr.update(5000, 200_000)
    before = est.ctr.trials
    est.ctr.discount()
    assert est.ctr.trials < before * 0.2


def test_paused_channel_is_evacuated(demo_plan, catalog, curves):
    summary = run_campaign(demo_plan, catalog, curves, _cfg("adaptive", "channel_pause"))
    caps_before = np.mean([h.caps["programmatic"] for h in summary.hours[200:240]])
    caps_after = np.mean([h.caps["programmatic"] for h in summary.hours[260:300]])
    assert caps_after < caps_before * 0.5
    assert any(p.from_channel == "programmatic" for p in summary.proposals)


def test_adaptive_beats_static_without_shock(demo_plan, catalog, curves):
    """Без шока: суммарное отклонение от плана (расход + KPI) меньше, чем у заморозки, и оба ниже 20 %."""
    stats = compare_strategies(demo_plan, catalog, curves, ("static", "adaptive"), "stable", seeds=SEEDS)
    a, s = stats["adaptive"].mean, stats["static"].mean
    assert a["final_deviation_kpi"] < s["final_deviation_kpi"]
    assert a["final_deviation_spend"] + a["final_deviation_kpi"] < s["final_deviation_spend"] + s["final_deviation_kpi"]
    assert a["final_deviation_spend"] <= 0.20 and a["final_deviation_kpi"] <= 0.20


def test_adaptive_closer_to_plan_after_shock(demo_plan, catalog, curves):
    for scenario in ("ctr_drop", "cpm_spike", "channel_pause"):
        stats = compare_strategies(demo_plan, catalog, curves, ("static", "adaptive"), scenario, seeds=SEEDS)
        a, s = stats["adaptive"].mean, stats["static"].mean
        assert a["final_deviation_kpi"] < s["final_deviation_kpi"], scenario
        assert a["final_deviation_spend"] + a["final_deviation_kpi"] < s["final_deviation_spend"] + s["final_deviation_kpi"], scenario
        assert a["final_deviation_kpi"] <= 0.20 and a["final_deviation_spend"] <= 0.20, scenario


def test_proposals_carry_two_prices(demo_plan, catalog, curves):
    summary = run_campaign(demo_plan, catalog, curves, _cfg("adaptive", "cpm_spike"))
    assert summary.proposals
    for p in summary.proposals:
        assert p.amount_rub > 0 and p.cause and p.cost_of_decision and p.cost_of_inaction
        assert p.applied_by in ("system", "human", "pending")


def test_automation_limit_blocks_large_moves(demo_plan, catalog, curves):
    limited = demo_plan.model_copy(update={"brief": demo_plan.brief.model_copy(update={"automation_limit_rub": 10_000.0})})
    cfg = _cfg("adaptive", "channel_pause")
    cfg.auto_apply_above_limit = False
    summary = run_campaign(limited, catalog, curves, cfg)
    assert summary.human_requests > 0
    assert any(p.applied_by == "pending" for p in summary.proposals)


def test_human_approval_applies_move_on_same_random_numbers(demo_plan, catalog, curves):
    """Человек в контуре: одобренный ход применяется, отклонённый остаётся ожидающим.

    Оба прогона идут на одних зёрнах (общие случайные числа), поэтому разница
    итогов это цена решения по факту, а не оценка.
    """
    limited = demo_plan.model_copy(update={"brief": demo_plan.brief.model_copy(update={"automation_limit_rub": 10_000.0})})
    cfg = _cfg("adaptive", "channel_pause")
    cfg.auto_apply_above_limit = False
    before = run_campaign(limited, catalog, curves, cfg)
    pending = [p for p in before.proposals if p.applied_by == "pending"]
    assert pending
    cfg.approved_hours = (pending[0].hour,)
    after = run_campaign(limited, catalog, curves, cfg)
    approved = [p for p in after.proposals if p.hour == pending[0].hour]
    assert approved and approved[0].applied_by == "human"
    assert after.human_requests > 0
    assert after.actual_kpi != before.actual_kpi or after.actual_spend != before.actual_spend
