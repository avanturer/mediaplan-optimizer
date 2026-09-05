"""Веб-кабинет MediaPlan Optimizer: FastAPI поверх harness.

Один процесс держит в памяти публичный каталог, ретро-историю и кривые
(считаются при старте за секунду), планы по их идентификаторам и результаты
прогонов. Кабинет ничего не считает сам: планировщик и исполнитель живут в
``brain``, мир в ``world``, стыкует их ``harness``.

Правила API, важные для показа: любая ошибка возвращается как JSON
``{"detail": "..."}`` по-русски с кодом 4xx, а не как 500 text/plain; прогон
возможен только для утверждённого и достижимого плана; шок из интерфейса
проверяется на канал плана и горизонт кампании.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, model_validator

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
from harness.compare import compare_strategies
from harness.retro import collect_retro_history
from harness.runner import RunConfig, run_campaign
from world import SCENARIOS, build_catalog

STATIC_DIR = Path(__file__).resolve().parent / "static"
CASE_DEVIATION_THRESHOLD = 0.20  # порог приёмки кейса: отклонение в конце не более 20 %
MAX_RUNS_IN_MEMORY = 50  # прогон весит около мегабайта; каждое «Перенести» создаёт новый
MAX_HORIZON_DAYS = 30  # кейс: 14–21 день; мир откалиброван на этот масштаб
MAX_CUSTOM_SHOCKS = 3  # своих шоков за прогон; harness принимает список любой длины

Strategy = Literal["static", "proportional_pacing", "pid", "adaptive"]
STRATEGY_TITLES = {
    "adaptive": "с перераспределением",
    "static": "без перераспределения (план буквально)",
    "pid": "ПИД-регулятор темпа",
    "proportional_pacing": "пропорциональный темп",
}

# Пресеты каналов (сценарий продукта: «несколько пресетов на ваш выбор»).
PRESETS: dict[str, dict[str, Any]] = {
    "all": {"title": "Все восемь каналов", "channels": ["social_1", "social_2", "social_3", "programmatic", "marketplace_1", "marketplace_2", "marketplace_3", "sms"]},
    "performance": {"title": "Перформанс: programmatic и маркетплейсы", "channels": ["programmatic", "marketplace_1", "marketplace_2", "marketplace_3"]},
    "social_sms": {"title": "Соцсети и SMS", "channels": ["social_1", "social_2", "social_3", "sms"]},
    "narrow": {"title": "Узкий: две соцсети, маркетплейс, SMS", "channels": ["social_2", "social_3", "marketplace_1", "sms"]},
}

SCENARIO_TITLES = {
    "stable": "спокойный рынок",
    "ctr_drop": "CTR −40 % в крупном маркетплейсе",
    "cpm_spike": "CPM ×2 в крупном маркетплейсе",
    "cpm_spike_recovery": "CPM ×1.4 на двое суток с восстановлением",
    "cvr_drop": "CR −40 % в маркетплейсе",
    "capacity_cut": "база SMS сжалась вдвое на четверо суток",
    "channel_pause": "programmatic выключен на трое суток",
    "demand_surge": "всплеск спроса в соцсетях на двое суток",
}

BINDING_TITLES = {
    "capacity": "ёмкость каналов",
    "horizon": "срок кампании",
    "economics": "цена результата",
    "channel_set": "набор каналов",
}

# Русские подписи для ошибок валидации: человек видит поле и правило, а не путь pydantic.
RU_FIELDS = {
    "budget_rub": "бюджет",
    "horizon_days": "горизонт",
    "target_value": "целевой объём",
    "target_kpi": "KPI",
    "channel_ids": "набор каналов",
    "max_cpa_rub": "потолок CPA",
    "automation_limit_rub": "лимит автоматики",
    "multiplier": "сила шока",
    "start_hour": "час шока",
    "duration_hours": "длительность шока",
    "world_seed": "зерно мира",
    "noise_seed": "зерно шума",
    "seeds": "число миров",
    "strategy": "стратегия",
    "scenario_id": "сценарий",
    "parameter": "параметр шока",
    "channel_id": "канал",
    "locked": "фиксация канала",
    "plan_id": "план",
    "hour": "час",
    "decision": "решение",
    "mode": "постановка",
    "objective": "что максимизируем",
    "shocks": "свои шоки",
}
RU_RULES = {
    "greater_than": "должно быть больше {gt}",
    "greater_than_equal": "не меньше {ge}",
    "less_than": "должно быть меньше {lt}",
    "less_than_equal": "не больше {le}",
    "missing": "не заполнено",
    "literal_error": "недопустимое значение",
    "enum": "недопустимое значение",
    "int_parsing": "нужно целое число",
    "float_parsing": "нужно число",
    "int_type": "нужно целое число",
    "float_type": "нужно число",
    "too_short": "список пуст",
    "too_long": "не больше {max_length}",
    "string_pattern_mismatch": "недопустимое значение",
}


def _rub(value: float) -> str:
    """Рубли с пробелом-разделителем тысяч, как принято в русской типографике."""
    return f"{value:,.0f}".replace(",", " ") + " ₽"


def _ru_errors(errors: list[dict[str, Any]]) -> str:
    parts = []
    for err in errors:
        loc = [str(x) for x in err.get("loc", ()) if x != "body"]
        field = loc[-1] if loc else ""
        name = RU_FIELDS.get(field, field)
        ctx = err.get("ctx") or {}
        rule = RU_RULES.get(err.get("type", ""), "")
        try:
            text = rule.format(**ctx) if rule else str(err.get("msg", "проверьте значение"))
        except (KeyError, IndexError):
            text = rule
        parts.append(f"{name}: {text}" if name else text)
    return "; ".join(parts) or "проверьте значения полей"


class State:
    catalog: PublicCatalog
    curves: dict[str, ResponseCurve]
    plans: dict[str, MediaPlan] = {}
    approved: dict[str, int] = {}  # plan_id → версия утверждения
    runs: dict[str, dict[str, Any]] = {}


state = State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.catalog = build_catalog(0)
    history = collect_retro_history(state.catalog)
    state.curves = build_curves(history, state.catalog)
    yield


app = FastAPI(title="MediaPlan Optimizer", version="0.3.0", lifespan=lifespan)


# --------------------------------------------------------- ошибки → JSON


@app.exception_handler(RequestValidationError)
async def _on_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _ru_errors(exc.errors())})


@app.exception_handler(ValidationError)
async def _on_model_validation(_: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _ru_errors(exc.errors())})


@app.exception_handler(ValueError)
async def _on_value_error(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(KeyError)
async def _on_key_error(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc).strip("'\"")})


# ------------------------------------------------------------------ схемы API


class BriefRequest(BaseModel):
    mode: Literal["A", "B"]
    preset: str = "all"
    channel_ids: list[str] | None = Field(default=None, min_length=1)
    budget_rub: float | None = None
    target_kpi: str | None = None
    target_value: float | None = None
    objective: str = "max_conversions"
    horizon_days: int = Field(default=21, ge=1, le=MAX_HORIZON_DAYS)
    max_cpa_rub: float | None = None
    locked: dict[str, float] = Field(default_factory=dict)
    automation_limit_rub: float | None = None


class ShockRequest(BaseModel):
    channel_id: str
    parameter: ShockParameter = ShockParameter.CTR
    multiplier: float = Field(default=0.6, gt=0)
    start_hour: int = Field(default=240, ge=0)
    duration_hours: int | None = Field(default=None, ge=1)


class ShocksMixin(BaseModel):
    """Свои шоки списком; старое одиночное поле `shock` принимается и переносится в список."""

    shock: ShockRequest | None = None
    shocks: list[ShockRequest] = Field(default_factory=list, max_length=MAX_CUSTOM_SHOCKS)

    @model_validator(mode="after")
    def _merge_single_shock(self):
        if self.shock is not None and not self.shocks:
            self.shocks = [self.shock]
        if len(self.shocks) > MAX_CUSTOM_SHOCKS:
            raise ValueError(f"своих шоков не больше {MAX_CUSTOM_SHOCKS}")
        return self


class RunRequest(ShocksMixin):
    plan_id: str
    strategy: Strategy = "adaptive"
    scenario_id: str = "stable"
    world_seed: int = 1
    noise_seed: int | None = Field(default=None, description="по умолчанию 10000 + зерно мира")
    auto_apply_above_limit: bool = False
    hold_plan: bool = True
    approved_hours: list[int] = Field(default_factory=list)


class DecideRequest(BaseModel):
    hour: int
    decision: Literal["approve", "decline"]


class DegradationRequest(BaseModel):
    plan_id: str
    world_seed: int = 1
    channel_id: str = "marketplace_1"
    parameter: ShockParameter = ShockParameter.CTR
    start_hour: int = Field(default=240, ge=0)
    multipliers: list[float] = Field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2])
    hold_plan: bool = True
    auto_apply_above_limit: bool = False


class CompareRequest(ShocksMixin):
    plan_id: str
    scenario_id: str = "stable"
    seeds: int = Field(default=20, ge=2, le=30)
    hold_plan: bool = True


class StressRequest(BaseModel):
    plan_id: str
    world_seed: int = 1
    hold_plan: bool = True
    auto_apply_above_limit: bool = False


# --------------------------------------------------------------- endpoints


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "plans": len(state.plans), "runs": len(state.runs)}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "catalog": state.catalog.model_dump(mode="json"),
        "presets": PRESETS,
        "scenarios": {k: {**v.model_dump(mode="json"), "title": SCENARIO_TITLES.get(k, k)} for k, v in SCENARIOS.items()},
        "strategies": list(STRATEGY_TITLES),
        "strategy_titles": STRATEGY_TITLES,
        "binding_titles": BINDING_TITLES,
        "shock_parameters": [p.value for p in ShockParameter],
        "case_threshold": CASE_DEVIATION_THRESHOLD,
        "max_horizon_days": MAX_HORIZON_DAYS,
    }


@app.post("/api/plan")
def make_plan(req: BriefRequest) -> dict[str, Any]:
    if req.channel_ids is None and req.preset not in PRESETS:
        raise HTTPException(422, f"неизвестный пресет «{req.preset}»; доступны: {', '.join(PRESETS)}")
    channels = req.channel_ids or PRESETS[req.preset]["channels"]
    unknown = [c for c in channels if c not in state.catalog.channel_ids]
    if unknown:
        raise HTTPException(422, f"каналов нет в каталоге: {', '.join(unknown)}")
    outside = [c for c in req.locked if c not in channels]
    if outside:
        raise HTTPException(422, f"зафиксирован канал вне набора: {', '.join(outside)}")
    if req.mode == "A":
        if req.budget_rub is None:
            raise HTTPException(422, "бюджет не задан")
        locked_sum = sum(req.locked.values())
        if locked_sum > req.budget_rub:
            raise HTTPException(422, f"зафиксировано {_rub(locked_sum)}, это больше бюджета {_rub(req.budget_rub)}")
    elif req.target_value is None:
        raise HTTPException(422, "целевой объём не задан")

    payload: dict[str, Any] = {
        "objective": req.objective,
        "horizon_days": req.horizon_days,
        "channel_ids": channels,
        "max_cpa_rub": req.max_cpa_rub,
        "locked": req.locked,
        "automation_limit_rub": req.automation_limit_rub,
    }
    if req.mode == "A":
        payload["budget_rub"] = req.budget_rub
    else:
        payload["target_kpi"] = req.target_kpi
        payload["target_value"] = req.target_value
    brief = Brief(**payload)
    media_plan = build_plan(brief, state.catalog, state.curves)
    state.plans[media_plan.plan_id] = media_plan
    return _plan_view(media_plan)


@app.get("/api/plan/{plan_id}")
def get_plan(plan_id: str) -> dict[str, Any]:
    return _plan_view(_plan(plan_id))


@app.post("/api/plan/{plan_id}/approve")
def approve_plan(plan_id: str) -> dict[str, Any]:
    media_plan = _plan(plan_id)
    _check_plan_usable(media_plan)
    # план неизменяем по id, поэтому утверждение идемпотентно: версия одна
    state.approved.setdefault(plan_id, 1)
    return {"plan_id": plan_id, "approved_version": state.approved[plan_id]}


def _check_plan_usable(media_plan: MediaPlan) -> None:
    """Достижимость и непустота проверяются раньше утверждения: иначе совет «утвердите план» бессмыслен."""
    if not media_plan.is_feasible:
        raise HTTPException(409, "план недостижим: примените один из предложенных ходов и утвердите новый план")
    if media_plan.total_budget_rub <= 0 or media_plan.total_kpi <= 0:
        raise HTTPException(409, "план пуст: ни один канал не проходит потолок CPA — поднимите потолок или уберите его")


@app.post("/api/run")
def run(req: RunRequest) -> dict[str, Any]:
    return _run(req)


def _run(req: RunRequest, decisions: dict[int, str] | None = None) -> dict[str, Any]:
    media_plan = _plan(req.plan_id)
    _check_plan_usable(media_plan)
    if req.plan_id not in state.approved:
        raise HTTPException(409, "план не утверждён: нажмите «Утвердить план»")
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(422, f"неизвестный сценарий «{req.scenario_id}»")
    injected = [_shock(s, media_plan) for s in req.shocks]
    noise_seed = req.noise_seed if req.noise_seed is not None else 10_000 + req.world_seed
    seeds = SeedBundle(catalog_seed=0, world_seed=req.world_seed, noise_seed=noise_seed)
    main = run_campaign(
        media_plan, state.catalog, state.curves,
        RunConfig(
            req.strategy, req.scenario_id, seeds, injected, req.auto_apply_above_limit,
            hold_plan=req.hold_plan, approved_hours=tuple(req.approved_hours),
        ),
    )
    twin = main if req.strategy == "static" else run_campaign(media_plan, state.catalog, state.curves, RunConfig("static", req.scenario_id, seeds, injected))
    run_id = uuid.uuid4().hex[:8]
    view = {
        "run_id": run_id,
        "plan_id": req.plan_id,
        "approved_version": state.approved[req.plan_id],
        "scenario_id": req.scenario_id,
        "scenario_title": SCENARIO_TITLES.get(req.scenario_id, req.scenario_id),
        "strategy_title": STRATEGY_TITLES.get(req.strategy, req.strategy),
        "custom_shocks": [s.model_dump(mode="json") for s in req.shocks],
        "seeds": seeds.model_dump(mode="json"),
        "main": _run_view(main),
        "frozen": _twin_view(twin),
        "verdict": _verdict(media_plan, main, twin),
        "decisions": decisions or {},
    }
    _remember_run(run_id, view, req)
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
        raise HTTPException(status_code=404, detail="прогон не найден: запустите кампанию заново") from exc
    prev_view, prev_req = stored["view"], stored["request"]
    hours_with_cards = {p["hour"] for p in prev_view["main"]["proposals"]}
    if req.hour not in hours_with_cards:
        raise HTTPException(422, f"в час {req.hour} карточки хода нет")
    decisions = {int(k): v for k, v in prev_view["decisions"].items()}
    if req.decision == "decline" and req.hour in prev_req.approved_hours:
        raise HTTPException(422, f"ход часа {req.hour} уже применён по вашему решению; отменить его можно только новым прогоном")
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
    _check_plan_usable(media_plan)
    _check_shock_target(req.channel_id, req.start_hour, media_plan)
    seeds = SeedBundle(catalog_seed=0, world_seed=req.world_seed, noise_seed=10_000 + req.world_seed)
    rows = []
    for mult in req.multipliers:
        if mult <= 0:
            raise HTTPException(422, f"сила шока должна быть больше нуля, получено {mult}")
        injected = [] if abs(mult - 1.0) < 1e-9 else [_shock(ShockRequest(channel_id=req.channel_id, parameter=req.parameter, multiplier=mult, start_hour=req.start_hour), media_plan)]
        adaptive = run_campaign(media_plan, state.catalog, state.curves, RunConfig("adaptive", "stable", seeds, injected, req.auto_apply_above_limit, hold_plan=req.hold_plan))
        frozen = run_campaign(media_plan, state.catalog, state.curves, RunConfig("static", "stable", seeds, injected))
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
    return {"plan_id": req.plan_id, "channel_id": req.channel_id, "parameter": req.parameter.value, "threshold": CASE_DEVIATION_THRESHOLD, "rows": rows}


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict[str, Any]:
    media_plan = _plan(req.plan_id)
    _check_plan_usable(media_plan)
    if req.scenario_id not in SCENARIOS:
        raise HTTPException(422, f"неизвестный сценарий «{req.scenario_id}»")
    injected = [_shock(s, media_plan) for s in req.shocks]
    stats = compare_strategies(media_plan, state.catalog, state.curves, scenario_id=req.scenario_id, seeds=req.seeds, injected=injected, hold_plan=req.hold_plan)
    out: dict[str, Any] = {}
    for name, st in stats.items():
        per_run = [
            {
                "world_seed": r.world_seed,
                "final_deviation_spend": r.final_deviation_spend,
                "final_deviation_kpi": r.final_deviation_kpi,
                "actual_kpi": r.actual_kpi,
                "actual_spend": r.actual_spend,
            }
            for r in st.runs
        ]
        within = sum(1 for r in st.runs if max(r.final_deviation_kpi, r.final_deviation_spend) <= CASE_DEVIATION_THRESHOLD)
        out[name] = {**st.to_dict(), "title": STRATEGY_TITLES.get(name, name), "per_run": per_run, "within_threshold_count": within}
    return {"threshold": CASE_DEVIATION_THRESHOLD, "seeds": req.seeds, "hold_plan": req.hold_plan, "strategies": out}


@app.post("/api/stress")
def stress(req: StressRequest) -> dict[str, Any]:
    """Стресс-тест плана до запуска: все сценарии шоков на одном мире, наша стратегия против заморозки."""
    media_plan = _plan(req.plan_id)
    _check_plan_usable(media_plan)
    seeds = SeedBundle(catalog_seed=0, world_seed=req.world_seed, noise_seed=10_000 + req.world_seed)
    rows = []
    for scenario_id in SCENARIOS:
        adaptive = run_campaign(media_plan, state.catalog, state.curves, RunConfig("adaptive", scenario_id, seeds, [], req.auto_apply_above_limit, hold_plan=req.hold_plan))
        frozen = run_campaign(media_plan, state.catalog, state.curves, RunConfig("static", scenario_id, seeds))
        rows.append(
            {
                "scenario_id": scenario_id,
                "title": SCENARIO_TITLES.get(scenario_id, scenario_id),
                "is_shock": scenario_id != "stable",
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


def _plan(plan_id: str) -> MediaPlan:
    try:
        return state.plans[plan_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="план не найден: рассчитайте его заново") from exc


def _check_shock_target(channel_id: str, start_hour: int, media_plan: MediaPlan) -> None:
    plan_channels = {a.channel_id for a in media_plan.allocations}
    if channel_id not in plan_channels:
        raise HTTPException(422, f"канала «{channel_id}» нет в плане: шок можно задать только по каналу плана")
    horizon_hours = len(media_plan.trajectory)
    if start_hour >= horizon_hours:
        raise HTTPException(422, f"шок с часа {start_hour} за пределами кампании: в ней {horizon_hours} часов")


def _shock(req: ShockRequest, media_plan: MediaPlan) -> ShockEvent:
    _check_shock_target(req.channel_id, req.start_hour, media_plan)
    return ShockEvent(
        start_hour=req.start_hour,
        duration_hours=req.duration_hours,
        target_channels=[req.channel_id],
        parameter=req.parameter,
        multiplier=req.multiplier,
    )


def _remember_run(run_id: str, view: dict[str, Any], req: RunRequest) -> None:
    state.runs[run_id] = {"view": view, "request": req}
    while len(state.runs) > MAX_RUNS_IN_MEMORY:
        state.runs.pop(next(iter(state.runs)))


def _plan_view(media_plan: MediaPlan) -> dict[str, Any]:
    data = media_plan.model_dump(mode="json")
    data["trajectory"] = [
        {k: v for k, v in point.items() if k != "by_channel_cum_spend_rub"} for point in data["trajectory"]
    ]
    data.pop("hourly_caps", None)
    data["total_budget_rub"] = media_plan.total_budget_rub
    data["total_kpi"] = media_plan.total_kpi
    data["approved_version"] = state.approved.get(media_plan.plan_id, 0)
    if media_plan.infeasibility is not None:
        # ходы — от дешёвого к дорогому, чтобы первым читался самый выгодный
        suggestions = data["infeasibility"]["suggestions"]
        suggestions.sort(key=lambda s: s["expected_budget_rub"])
        for s in suggestions:
            # планировщик предлагает сроки без оглядки на калибровку мира; кабинет не примет горизонт длиннее MAX_HORIZON_DAYS
            too_long = s["changed_field"] == "horizon_days" and s["suggested_value"] is not None and s["suggested_value"] > MAX_HORIZON_DAYS
            s["applicable"] = not too_long
            s["why_not"] = f"дольше {MAX_HORIZON_DAYS} дней: за пределами калибровки мира" if too_long else None
        data["infeasibility"]["binding_title"] = BINDING_TITLES.get(media_plan.infeasibility.binding_constraint.value, media_plan.infeasibility.binding_constraint.value)
    data["is_empty"] = media_plan.is_feasible and (media_plan.total_budget_rub <= 0 or media_plan.total_kpi <= 0)
    return data


def _run_view(summary: RunSummary) -> dict[str, Any]:
    data = summary.model_dump(mode="json")
    data["strategy_title"] = STRATEGY_TITLES.get(summary.strategy, summary.strategy)
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
        }
        for h in data["hours"]
    ]
    return data


def _twin_view(twin: RunSummary) -> dict[str, Any]:
    """Двойник-заморозка: только то, что рисует кабинет, без мегабайтов почасовых таблиц."""
    return {
        "strategy": twin.strategy,
        "actual_kpi": twin.actual_kpi,
        "actual_spend": twin.actual_spend,
        "final_deviation_kpi": twin.final_deviation_kpi,
        "final_deviation_spend": twin.final_deviation_spend,
        "hours": [{"hour": h.hour, "fact_cum_spend": h.fact_cum_spend, "fact_cum_kpi": h.fact_cum_kpi} for h in twin.hours],
    }


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
        "frozen_within_threshold": max(twin.final_deviation_kpi, twin.final_deviation_spend) <= CASE_DEVIATION_THRESHOLD,
        "threshold": CASE_DEVIATION_THRESHOLD,
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
