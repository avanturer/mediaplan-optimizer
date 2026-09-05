"""Веб-кабинет MediaPlan Optimizer: FastAPI поверх harness.

Один процесс держит в памяти публичный каталог, ретро-историю и кривые
(считаются при старте за секунду), планы по их идентификаторам и результаты
прогонов. Кабинет ничего не считает сам: планировщик и исполнитель живут в
``brain``, мир в ``world``, стыкует их ``harness``.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from brain.curves import ResponseCurve, build_curves
from brain.planner import plan as build_plan
from contracts import (
    Brief,
    MediaPlan,
    PublicCatalog,
    RunSummary,
    SeedBundle,
    ShockEvent,
    ShockParameter,
)
from contracts.targeting import AudienceTargeting
from harness.compare import compare_strategies
from harness.retro import collect_retro_history
from harness.runner import RunConfig, run_campaign
from world import SCENARIOS, build_catalog
from world.settings import WorldSettings
from world.targeting import catalog_for_targeting

STATIC_DIR = Path(__file__).resolve().parent / "static"
CASE_DEVIATION_THRESHOLD = 0.20  # порог приёмки кейса: отклонение в конце не более 20 %

# Пресеты каналов (сценарий продукта: «несколько пресетов на ваш выбор»).
PRESETS: dict[str, dict[str, Any]] = {
    "all": {"title": "Все восемь каналов", "channels": ["social_1", "social_2", "social_3", "programmatic", "marketplace_1", "marketplace_2", "marketplace_3", "sms"]},
    "performance": {"title": "Перформанс: programmatic и маркетплейсы", "channels": ["programmatic", "marketplace_1", "marketplace_2", "marketplace_3"]},
    "social_sms": {"title": "Соцсети и SMS", "channels": ["social_1", "social_2", "social_3", "sms"]},
    "narrow": {"title": "Узкий пресет: две соцсети, один маркетплейс, SMS", "channels": ["social_2", "social_3", "marketplace_1", "sms"]},
}

SCENARIO_TITLES = {
    "stable": "спокойный рынок",
    "fraud_surge": "бот-ферма в programmatic: CTR растёт, конверсии падают",
    "sms_weekly_limit": "недельная квота SMS снижена вдвое",
    "ctr_drop": "CTR −40 % в крупном маркетплейсе",
    "cpm_spike": "CPM ×2 в крупном маркетплейсе",
    "cpm_spike_recovery": "CPM ×1.4 на двое суток с восстановлением",
    "cvr_drop": "CR −40 % в маркетплейсе",
    "capacity_cut": "база SMS сжалась вдвое на четверо суток",
    "channel_pause": "programmatic выключен на трое суток",
    "demand_surge": "всплеск спроса в соцсетях на двое суток",
}


class State:
    catalog: PublicCatalog
    curves: dict[str, ResponseCurve]
    plans: dict[str, MediaPlan] = {}
    approved: dict[str, int] = {}  # plan_id → версия утверждения
    runs: dict[str, dict[str, Any]] = {}


state = State()
app = FastAPI(title="MediaPlan Optimizer", version="0.2.0")


@app.on_event("startup")
def _startup() -> None:
    state.catalog = build_catalog(0)
    history = collect_retro_history(state.catalog)
    state.curves = build_curves(history, state.catalog)


# ------------------------------------------------------------------ схемы API


class BriefRequest(BaseModel):
    mode: str = Field(pattern="^(A|B)$")
    preset: str = "all"
    channel_ids: list[str] | None = None
    budget_rub: float | None = None
    target_kpi: str | None = None
    target_value: float | None = None
    objective: str = "max_conversions"
    horizon_days: int = 21
    max_cpa_rub: float | None = None
    locked: dict[str, float] = Field(default_factory=dict)
    automation_limit_rub: float | None = None
    targeting: AudienceTargeting = Field(default_factory=AudienceTargeting)


class ShockRequest(BaseModel):
    channel_id: str
    parameter: str = "ctr"
    multiplier: float = 0.6
    start_hour: int = 240
    duration_hours: int | None = None
    recovery: str = "none"


class RunRequest(BaseModel):
    plan_id: str
    strategy: str = "adaptive"
    scenario_id: str = "stable"
    world_seed: int = 1
    noise_seed: int = 10001
    shock: ShockRequest | None = None
    auto_apply_above_limit: bool = True
    hold_plan: bool = True
    approved_hours: list[int] = Field(default_factory=list)
    world_settings: WorldSettings | None = None


class DecideRequest(BaseModel):
    hour: int
    decision: str = Field(pattern="^(approve|decline)$")


class DegradationRequest(BaseModel):
    plan_id: str
    world_seed: int = 1
    channel_id: str = "marketplace_1"
    parameter: str = "ctr"
    start_hour: int = 240
    multipliers: list[float] = Field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2])
    hold_plan: bool = True


class CompareRequest(BaseModel):
    plan_id: str
    scenario_id: str = "stable"
    seeds: int = 20
    shock: ShockRequest | None = None
    hold_plan: bool = True
    world_settings: WorldSettings | None = None


class StressRequest(BaseModel):
    plan_id: str
    world_seed: int = 1
    hold_plan: bool = True


# --------------------------------------------------------------- endpoints


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "catalog": state.catalog.model_dump(mode="json"),
        "presets": PRESETS,
        "scenarios": {k: {**v.model_dump(mode="json"), "title": SCENARIO_TITLES.get(k, k)} for k, v in SCENARIOS.items()},
        "strategies": ["static", "proportional_pacing", "pid", "adaptive"],
        "shock_parameters": [p.value for p in ShockParameter],
        "case_threshold": CASE_DEVIATION_THRESHOLD,
    }


@app.post("/api/plan")
def make_plan(req: BriefRequest) -> dict[str, Any]:
    channels = req.channel_ids or PRESETS.get(req.preset, PRESETS["all"])["channels"]
    payload: dict[str, Any] = {
        "objective": req.objective,
        "horizon_days": req.horizon_days,
        "channel_ids": channels,
        "max_cpa_rub": req.max_cpa_rub,
        "locked": req.locked,
        "automation_limit_rub": req.automation_limit_rub,
        "targeting": req.targeting,
    }
    if req.mode == "A":
        payload["budget_rub"] = req.budget_rub
    else:
        payload["target_kpi"] = req.target_kpi
        payload["target_value"] = req.target_value
    try:
        brief = Brief(**payload)
        catalog, curves = _target_context(brief.targeting.model_dump_json())
        media_plan = build_plan(brief, catalog, curves)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state.plans[media_plan.plan_id] = media_plan
    return _plan_view(media_plan)


@app.get("/api/plan/{plan_id}")
def get_plan(plan_id: str) -> dict[str, Any]:
    return _plan_view(_plan(plan_id))


@app.post("/api/plan/{plan_id}/approve")
def approve_plan(plan_id: str) -> dict[str, Any]:
    _plan(plan_id)
    state.approved[plan_id] = state.approved.get(plan_id, 0) + 1
    return {"plan_id": plan_id, "approved_version": state.approved[plan_id]}


@app.post("/api/run")
def run(req: RunRequest) -> dict[str, Any]:
    try:
        return _run(req)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run(req: RunRequest, decisions: dict[int, str] | None = None) -> dict[str, Any]:
    media_plan = _plan(req.plan_id)
    catalog, curves = _context(media_plan)
    injected = [_shock(req.shock)] if req.shock else []
    seeds = SeedBundle(catalog_seed=0, world_seed=req.world_seed, noise_seed=req.noise_seed)
    main = run_campaign(
        media_plan, catalog, curves,
        RunConfig(
            req.strategy, req.scenario_id, seeds, injected, req.auto_apply_above_limit,
            hold_plan=req.hold_plan, approved_hours=tuple(req.approved_hours),
            world_settings=req.world_settings,
        ),
    )
    twin = main if req.strategy == "static" else run_campaign(media_plan, catalog, curves, RunConfig("static", req.scenario_id, seeds, injected, world_settings=req.world_settings))
    run_id = uuid.uuid4().hex[:8]
    view = {
        "run_id": run_id,
        "main": _run_view(main),
        "frozen": _run_view(twin),
        "verdict": _verdict(media_plan, main, twin),
        "decisions": decisions or {},
    }
    state.runs[run_id] = {"view": view, "request": req}
    return view


@app.post("/api/run/{run_id}/decide")
def decide(run_id: str, req: DecideRequest) -> dict[str, Any]:
    """Человек в контуре: одобрить или отклонить ход выше лимита полномочий.

    Одобрение перепрогоняет ту же кампанию на тех же зёрнах (общие случайные
    числа), поэтому разница итогов это цена решения по факту, а не оценка.
    """
    try:
        stored = state.runs[run_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="прогон не найден") from exc
    prev_view, prev_req = stored["view"], stored["request"]
    decisions = {int(k): v for k, v in prev_view["decisions"].items()}
    decisions[req.hour] = req.decision
    if req.decision == "decline":
        prev_view["decisions"] = decisions
        return {**prev_view, "effect": None}
    new_req = prev_req.model_copy(update={"approved_hours": sorted(set(prev_req.approved_hours) | {req.hour})})
    view = _run(new_req, decisions)
    before, after = prev_view["verdict"], view["verdict"]
    view["effect"] = {
        "hour": req.hour,
        "kpi_delta": after["actual_kpi"] - before["actual_kpi"],
        "spend_delta": after["actual_spend"] - before["actual_spend"],
        "deviation_kpi_before": before["final_deviation_kpi"],
        "deviation_kpi_after": after["final_deviation_kpi"],
    }
    return view


@app.post("/api/degradation")
def degradation(req: DegradationRequest) -> dict[str, Any]:
    """Стенд честности: как растёт отклонение при усилении шока, где решение ломается."""
    media_plan = _plan(req.plan_id)
    catalog, curves = _context(media_plan)
    seeds = SeedBundle(catalog_seed=0, world_seed=req.world_seed, noise_seed=10_000 + req.world_seed)
    rows = []
    for mult in req.multipliers:
        injected = [] if abs(mult - 1.0) < 1e-9 else [_shock(ShockRequest(channel_id=req.channel_id, parameter=req.parameter, multiplier=mult, start_hour=req.start_hour))]
        adaptive = run_campaign(media_plan, catalog, curves, RunConfig("adaptive", "stable", seeds, injected, hold_plan=req.hold_plan))
        frozen = run_campaign(media_plan, catalog, curves, RunConfig("static", "stable", seeds, injected))
        rows.append(
            {
                "multiplier": mult,
                "adaptive_dev_kpi": adaptive.final_deviation_kpi,
                "adaptive_dev_spend": adaptive.final_deviation_spend,
                "frozen_dev_kpi": frozen.final_deviation_kpi,
                "frozen_dev_spend": frozen.final_deviation_spend,
                "adaptive_kpi": adaptive.actual_kpi,
                "frozen_kpi": frozen.actual_kpi,
                "detection_hour": adaptive.detection_hours.get(req.channel_id),
                "withstands": max(adaptive.final_deviation_kpi, adaptive.final_deviation_spend) <= CASE_DEVIATION_THRESHOLD,
            }
        )
    return {"plan_id": req.plan_id, "channel_id": req.channel_id, "parameter": req.parameter, "threshold": CASE_DEVIATION_THRESHOLD, "rows": rows}


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict[str, Any]:
    media_plan = _plan(req.plan_id)
    catalog, curves = _context(media_plan)
    injected = [_shock(req.shock)] if req.shock else []
    stats = compare_strategies(media_plan, catalog, curves, scenario_id=req.scenario_id, seeds=min(max(req.seeds, 2), 30), injected=injected,
                               world_settings=req.world_settings)
    return {name: st.to_dict() for name, st in stats.items()}


@app.post("/api/stress")
def stress(req: StressRequest) -> dict[str, Any]:
    """Стресс-тест плана до запуска: все сценарии шоков на одном мире, наша стратегия против заморозки."""
    media_plan = _plan(req.plan_id)
    catalog, curves = _context(media_plan)
    seeds = SeedBundle(catalog_seed=0, world_seed=req.world_seed, noise_seed=10_000 + req.world_seed)
    rows = []
    for scenario_id in SCENARIOS:
        adaptive = run_campaign(media_plan, catalog, curves, RunConfig("adaptive", scenario_id, seeds, hold_plan=req.hold_plan))
        frozen = run_campaign(media_plan, catalog, curves, RunConfig("static", scenario_id, seeds))
        rows.append(
            {
                "scenario_id": scenario_id,
                "title": SCENARIO_TITLES.get(scenario_id, scenario_id),
                "adaptive_dev_kpi": adaptive.final_deviation_kpi,
                "adaptive_dev_spend": adaptive.final_deviation_spend,
                "frozen_dev_kpi": frozen.final_deviation_kpi,
                "frozen_dev_spend": frozen.final_deviation_spend,
                "adaptive_kpi": adaptive.actual_kpi,
                "frozen_kpi": frozen.actual_kpi,
                "detection_hours": adaptive.detection_hours,
                "withstands": max(adaptive.final_deviation_kpi, adaptive.final_deviation_spend) <= CASE_DEVIATION_THRESHOLD,
                "moves": len(adaptive.proposals),
            }
        )
    return {"plan_id": req.plan_id, "world_seed": req.world_seed, "threshold": CASE_DEVIATION_THRESHOLD, "rows": rows}


# ----------------------------------------------------------------- helpers


@lru_cache(maxsize=32)
def _target_context(targeting_json: str) -> tuple[PublicCatalog, dict[str, ResponseCurve]]:
    targeting = AudienceTargeting.model_validate_json(targeting_json)
    if targeting == state.catalog.targeting:
        return state.catalog, state.curves
    catalog = catalog_for_targeting(state.catalog, targeting)
    history = collect_retro_history(catalog)
    return catalog, build_curves(history, catalog)


def _context(media_plan: MediaPlan) -> tuple[PublicCatalog, dict[str, ResponseCurve]]:
    return _target_context(media_plan.brief.targeting.model_dump_json())


def _plan(plan_id: str) -> MediaPlan:
    try:
        return state.plans[plan_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="план не найден") from exc


def _shock(req: ShockRequest) -> ShockEvent:
    return ShockEvent(
        start_hour=req.start_hour,
        duration_hours=req.duration_hours,
        target_channels=[req.channel_id],
        parameter=ShockParameter(req.parameter),
        multiplier=req.multiplier,
        recovery=req.recovery,
    )


def _plan_view(media_plan: MediaPlan) -> dict[str, Any]:
    data = media_plan.model_dump(mode="json")
    data["trajectory"] = [
        {k: v for k, v in point.items() if k != "by_channel_cum_spend_rub"} for point in data["trajectory"]
    ]
    data.pop("hourly_caps", None)
    data["total_budget_rub"] = media_plan.total_budget_rub
    data["total_kpi"] = media_plan.total_kpi
    data["approved_version"] = state.approved.get(media_plan.plan_id, 0)
    return data


def _run_view(summary: RunSummary) -> dict[str, Any]:
    data = summary.model_dump(mode="json")
    data["hours"] = [
        {
            "hour": h["hour"],
            "plan_cum_spend": h["plan_cum_spend"],
            "plan_cum_kpi": h["plan_cum_kpi"],
            "fact_cum_spend": h["fact_cum_spend"],
            "fact_cum_kpi": h["fact_cum_kpi"],
            "by_channel": h["fact_by_channel"],
            "caps": h["caps"],
            "status": h["status"],
            "events": h["events"],
            "err_spend": h["tracking_error_spend"],
            "err_kpi": h["tracking_error_kpi"],
            "reserve": h["reserve_rub"],
            "deduplicated_reach": h["deduplicated_reach"],
        }
        for h in data["hours"]
    ]
    return data


def _verdict(media_plan: MediaPlan, main: RunSummary, twin: RunSummary) -> dict[str, Any]:
    plan_cpa = media_plan.total_budget_rub / media_plan.total_kpi if media_plan.total_kpi else 0.0
    kpi_gain = main.actual_kpi - twin.actual_kpi
    returned = max(media_plan.total_budget_rub - main.actual_spend, 0.0)
    return {
        "kpi_name": media_plan.kpi_name,
        "promised_kpi": main.promised_kpi,
        "actual_kpi": main.actual_kpi,
        "frozen_kpi": twin.actual_kpi,
        "promised_spend": main.promised_spend,
        "actual_spend": main.actual_spend,
        "frozen_spend": twin.actual_spend,
        "returned_budget_rub": returned,
        "mape_spend": main.mape_spend,
        "mape_kpi": main.mape_kpi,
        "frozen_mape_spend": twin.mape_spend,
        "frozen_mape_kpi": twin.mape_kpi,
        "final_deviation_kpi": main.final_deviation_kpi,
        "frozen_final_deviation_kpi": twin.final_deviation_kpi,
        "final_deviation_spend": main.final_deviation_spend,
        "frozen_final_deviation_spend": twin.final_deviation_spend,
        "kpi_gain_vs_frozen": kpi_gain,
        "rub_saved_vs_frozen": kpi_gain * plan_cpa,
        "human_requests": main.human_requests,
        "detection_hours": main.detection_hours,
        "shock_hours": main.shock_hours,
        "runtime_seconds": main.runtime_seconds,
        "within_threshold": max(main.final_deviation_kpi, main.final_deviation_spend) <= CASE_DEVIATION_THRESHOLD,
        "threshold": CASE_DEVIATION_THRESHOLD,
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
