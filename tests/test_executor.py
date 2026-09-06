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
    assert est.ctr.trials > 0
    est.ctr.reset()
    assert est.ctr.trials == 0 and abs(est.ctr.value - curves["marketplace_1"].ctr) < 1e-9


def test_paused_channel_is_evacuated(demo_plan, catalog, curves):
    summary = run_campaign(demo_plan, catalog, curves, _cfg("adaptive", "channel_pause"))
    caps_before = np.mean([h.caps["programmatic"] for h in summary.hours[200:240]])
    caps_after = np.mean([h.caps["programmatic"] for h in summary.hours[260:300]])
    assert caps_after < caps_before * 0.5
    assert any(p.from_channel == "programmatic" for p in summary.proposals)


def test_adaptive_beats_static_without_shock(demo_plan, catalog, curves):
    """Без шока: худшее из двух отклонений (расход, KPI) меньше, чем у заморозки, и оба ниже 20 %.

    Метрика кейса это порог на каждое отклонение, поэтому сравниваем по худшему из двух,
    а не по сумме: по сумме удержание плана в щедром мире равно заморозке (запись 34)."""
    stats = compare_strategies(demo_plan, catalog, curves, ("static", "adaptive"), "stable", seeds=SEEDS)
    a, s = stats["adaptive"].mean, stats["static"].mean
    assert a["final_deviation_kpi"] < s["final_deviation_kpi"]
    assert max(a["final_deviation_spend"], a["final_deviation_kpi"]) < max(s["final_deviation_spend"], s["final_deviation_kpi"])
    assert a["final_deviation_spend"] <= 0.20 and a["final_deviation_kpi"] <= 0.20


def test_adaptive_closer_to_plan_after_shock(demo_plan, catalog, curves):
    for scenario in ("ctr_drop", "cpm_spike", "channel_pause"):
        stats = compare_strategies(demo_plan, catalog, curves, ("static", "adaptive"), scenario, seeds=SEEDS)
        a, s = stats["adaptive"].mean, stats["static"].mean
        assert a["final_deviation_kpi"] < s["final_deviation_kpi"], scenario
        assert max(a["final_deviation_spend"], a["final_deviation_kpi"]) < max(s["final_deviation_spend"], s["final_deviation_kpi"]), scenario
        assert a["final_deviation_kpi"] <= 0.20 and a["final_deviation_spend"] <= 0.20, scenario


def test_proposals_carry_two_prices(demo_plan, catalog, curves):
    summary = run_campaign(demo_plan, catalog, curves, _cfg("adaptive", "cpm_spike"))
    assert summary.proposals
    transfers = [p for p in summary.proposals if p.from_channel != p.to_channel]
    assert transfers
    for p in summary.proposals:
        assert p.cause and p.cost_of_decision and p.cost_of_inaction
        assert p.applied_by in ("system", "human", "pending", "rejected")
        assert p.cause_kind in ("fact", "drop", "rise", "pause")
        # карточка «держим» при сломе без переноса: сумма ноль, донор и получатель совпадают
        assert (p.amount_rub > 0) == (p.from_channel != p.to_channel)


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


def _limited(demo_plan, limit=10_000.0):
    return demo_plan.model_copy(update={"brief": demo_plan.brief.model_copy(update={"automation_limit_rub": limit})})


def test_kpi_detector_catches_cvr_drop(demo_plan, catalog, curves):
    """KPI кейса это конверсии: падение CR вдвое в крупном канале замечается по конверсиям на рубль.

    Падение на 40 % в маленьком канале (сценарий cvr_drop, marketplace_3) на 3σ
    неотличимо от шума за оставшиеся дни: там детектор честно молчит.
    """
    from contracts import ShockEvent, ShockParameter

    cfg = _cfg("adaptive")
    cfg.injected = [ShockEvent(start_hour=240, target_channels=["marketplace_1"], parameter=ShockParameter.CVR, multiplier=0.5)]
    summary = run_campaign(demo_plan, catalog, curves, cfg)
    fired = summary.detection_hours.get("marketplace_1")
    assert fired is not None and 240 < fired <= 240 + 120
    assert any("конверсии на рубль" in e for e in summary.hours[fired - 1].events)


def test_rise_detector_flags_ctr_jump(demo_plan, catalog, curves):
    """Скачок кликов вверх при прежней цене это аномалия, а не подарок: канал помечается, деньги в него не льются."""
    from contracts import ShockEvent, ShockParameter

    jump = ShockEvent(start_hour=240, duration_hours=None, target_channels=["marketplace_1"], parameter=ShockParameter.CTR, multiplier=1.8)
    cfg = _cfg("adaptive")
    cfg.injected = [jump]
    summary = run_campaign(demo_plan, catalog, curves, cfg)
    assert summary.detection_hours.get("marketplace_1", 10_000) > 240
    assert any("похоже на фрод" in e for e in summary.hours[summary.detection_hours["marketplace_1"] - 1].events)


def test_pause_is_a_breakdown(demo_plan, catalog, curves):
    summary = run_campaign(demo_plan, catalog, curves, _cfg("adaptive", "channel_pause"))
    assert "programmatic" in summary.detection_hours
    fired = summary.detection_hours["programmatic"]
    assert any(h.status == "fire" for h in summary.hours[fired : fired + 6])


def test_card_appears_at_detection_hour(demo_plan, catalog, curves):
    summary = run_campaign(demo_plan, catalog, curves, _cfg("adaptive", "ctr_drop"))
    fired = summary.detection_hours["marketplace_1"]
    cards = [p for p in summary.proposals if p.from_channel == "marketplace_1" and p.cause_kind == "drop"]
    assert cards and cards[0].hour <= fired + 1
    assert cards[0].inaction_kpi_shortfall_abs is not None


def test_pending_card_freezes_donor(demo_plan, catalog, curves):
    """Пока ход выше лимита ждёт человека, автоматика не добирает его частями ниже лимита."""
    cfg = _cfg("adaptive", "channel_pause")
    cfg.auto_apply_above_limit = False
    summary = run_campaign(_limited(demo_plan), catalog, curves, cfg)
    pending = [p for p in summary.proposals if p.applied_by == "pending"]
    assert pending
    first = pending[0]
    later = [p for p in summary.proposals if p.hour > first.hour and p.from_channel == first.from_channel and p.applied_by == "system"]
    assert not later, [(p.hour, p.amount_rub, p.cause_kind) for p in later]


def test_rejected_hours_revert_a_move(demo_plan, catalog, curves):
    """Откат: ход, который система применила, можно отменить, и деньги остаются у донора."""
    base = run_campaign(demo_plan, catalog, curves, _cfg("adaptive", "channel_pause"))
    applied = [p for p in base.proposals if p.applied_by == "system"]
    assert applied
    target = applied[0]
    cfg = _cfg("adaptive", "channel_pause")
    cfg.rejected_hours = (target.hour,)
    reverted = run_campaign(demo_plan, catalog, curves, cfg)
    same = [p for p in reverted.proposals if p.hour == target.hour]
    assert same and same[0].applied_by == "rejected"
    assert reverted.actual_spend != base.actual_spend or reverted.actual_kpi != base.actual_kpi


def test_no_scheduled_replan_in_first_day(demo_plan, catalog, curves):
    """Первые сутки исполняется утверждённый план: ни одной карточки автоматики до 24-го часа
    в спокойном мире (перерешение по событию разрешено, но без шока событий нет)."""
    summary = run_campaign(demo_plan, catalog, curves, _cfg("adaptive"))
    early = [p for p in summary.proposals if p.hour < 24]
    assert not early, [(p.hour, p.from_channel, p.to_channel, p.amount_rub) for p in early]


def test_status_follows_case_threshold(demo_plan, catalog, curves):
    from brain.config import CASE_THRESHOLD
    from brain.executor import make_executor
    from contracts import TrackingStatus

    ex = make_executor("adaptive", demo_plan, catalog, curves, demo_plan.total_budget_rub)
    ex.hour = 100
    assert ex._status(0.0, CASE_THRESHOLD / 2 - 0.01) == TrackingStatus.OK
    assert ex._status(0.0, CASE_THRESHOLD / 2 + 0.01) == TrackingStatus.WATCH
    assert ex._status(0.0, CASE_THRESHOLD + 0.01) == TrackingStatus.FIRE
    assert ex._status(-(CASE_THRESHOLD + 0.01), 0.0) == TrackingStatus.FIRE


def test_locked_channel_is_never_a_donor(demo_brief, catalog, curves):
    from brain.planner import plan

    locked = demo_brief.model_copy(update={"locked": {"sms": 300_000.0}})
    p = plan(locked, catalog, curves)
    summary = run_campaign(p, catalog, curves, _cfg("adaptive"))
    moved = [pr for pr in summary.proposals if pr.from_channel == "sms" or pr.to_channel == "sms"]
    assert not moved, [(pr.hour, pr.from_channel, pr.to_channel, pr.amount_rub) for pr in moved]


def test_reach_detector_silent_without_shock(demo_brief, catalog, curves):
    from brain.planner import plan
    from contracts import Objective

    reach_brief = demo_brief.model_copy(update={"objective": Objective.MAX_REACH})
    p = plan(reach_brief, catalog, curves)
    for seed in (1, 2):
        summary = run_campaign(p, catalog, curves, _cfg("static", world_seed=seed))
        assert not summary.detection_hours, summary.detection_hours


def test_reserve_accounts_for_concavity(demo_plan, catalog, curves):
    """Резерв уравнивает итоговые отклонения: при опережении в полтора раза выбранная доля s даёт
    ожидаемое перевыполнение, равное недорасходу, а не (r − 1)/(r + 1) от остатка."""
    from brain.config import RESERVE_ELASTICITY
    from brain.executor import make_executor

    ex = make_executor("adaptive", demo_plan, catalog, curves, demo_plan.total_budget_rub)
    ex.hour = 240
    _, plan_to_date = ex._plan_cum(ex.hour)
    _, plan_total = ex._plan_cum(ex.horizon)
    ratio = 1.5
    ex.fact_cum_kpi = ratio * plan_to_date
    remaining = 600_000.0
    use = ex._budget_to_use(None, remaining)
    s_share = 1 - use / remaining
    underspend = s_share * remaining / demo_plan.total_budget_rub
    overshoot = (ex.fact_cum_kpi + ratio * (plan_total - plan_to_date) * (1 - s_share) ** RESERVE_ELASTICITY) / plan_total - 1
    assert 0 < s_share < 1
    assert abs(underspend - overshoot) < 1e-3
    ex.fact_cum_kpi = plan_to_date  # без опережения резерв не нужен
    assert ex._budget_to_use(None, remaining) == remaining


def test_pending_card_not_reissued_on_pause(demo_plan, catalog, curves):
    """Пауза канала без автоприменения: одна карточка ждёт решения, а не новая каждые шесть часов."""
    from contracts import ShockEvent, ShockParameter

    plan_with_limit = demo_plan.model_copy(update={"brief": demo_plan.brief.model_copy(update={"automation_limit_rub": 10_000.0})})
    cfg = _cfg("adaptive")
    cfg.injected = [ShockEvent(target_channels=["programmatic"], parameter=ShockParameter.PAUSE, multiplier=1.0, start_hour=240, duration_hours=72)]
    cfg.auto_apply_above_limit = False
    summary = run_campaign(plan_with_limit, catalog, curves, cfg)
    pending = [p for p in summary.proposals if p.applied_by == "pending" and p.from_channel == "programmatic"]
    assert len(pending) == 1, [(p.hour, p.amount_rub) for p in pending]
    assert summary.human_requests == len([p for p in summary.proposals if p.applied_by == "pending"])
