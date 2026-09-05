"""Цикл прогона кампании: план → часовые решения → мир → факт.

Единственное место, где мозг и мир встречаются. Runner знает сценарий и
шоки для организации эксперимента, стратегия видит только наблюдения.
Шок из интерфейса добавляется через ``injected`` и попадает в мир в нужный
час через ``inject_shock``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from brain.curves import ResponseCurve
from brain.executor import BaseExecutor, make_executor
from brain.executor.controller import kpi_of_observation, spend_by_channel
from brain.ml import MLBundle
from contracts import (
    Action,
    HourRecord,
    MediaPlan,
    PublicCatalog,
    RunSummary,
    SeedBundle,
    ShockEvent,
)
from contracts.ml import MLConfig
from harness.metrics import coefficient_of_variation, final_deviation, mape, unsmoothness, wape
from world.settings import WorldSettings
from world.simulator import Simulator


@dataclass
class RunConfig:
    strategy: str = "adaptive"
    scenario_id: str = "stable"
    seeds: SeedBundle = field(default_factory=SeedBundle)
    injected: list[ShockEvent] = field(default_factory=list)
    auto_apply_above_limit: bool = True
    stop_at_first_event: bool = False
    hold_plan: bool = True  # adaptive: держать план (резерв) или выжимать максимум KPI
    approved_hours: tuple[int, ...] = ()  # ходы выше лимита, одобренные человеком (по часу карточки)
    world_settings: WorldSettings | None = None
    ml: MLConfig = field(default_factory=MLConfig)


def run_campaign(
    plan: MediaPlan,
    catalog: PublicCatalog,
    curves: dict[str, ResponseCurve],
    config: RunConfig,
    simulator: Simulator | None = None,
    ml_bundle: MLBundle | None = None,
) -> RunSummary:
    if not plan.is_feasible:
        raise ValueError("нельзя прогнать недостижимый план")
    started = time.perf_counter()
    if plan.catalog_id != catalog.catalog_id or plan.brief.targeting != catalog.targeting:
        raise ValueError("для исполнения требуется каталог выбранного сегмента, использованный в плане")
    if simulator is not None and config.world_settings is not None and simulator.settings != config.world_settings:
        raise ValueError("настройки переданного симулятора расходятся с RunConfig")
    sim = simulator or Simulator(catalog, settings=config.world_settings)
    horizon = len(plan.trajectory)
    total_budget = plan.total_budget_rub
    channel_ids = [a.channel_id for a in plan.allocations]
    kwargs = (
        {
            "auto_apply_above_limit": config.auto_apply_above_limit,
            "hold_plan": config.hold_plan,
            "approved_hours": set(config.approved_hours),
        }
        if config.strategy == "adaptive"
        else {}
    )
    executor: BaseExecutor = make_executor(config.strategy, plan, catalog, curves, total_budget,
                                          ml_config=config.ml, ml_bundle=ml_bundle, **kwargs)

    sim.reset(config.seeds, config.scenario_id, horizon_hours=horizon, total_budget=total_budget, channel_ids=channel_ids,
              targeting=plan.brief.targeting)
    for event in config.injected:
        sim.inject_shock(event)

    records: list[HourRecord] = []
    plan_spend = np.array([p.cum_spend_rub for p in plan.trajectory])
    plan_kpi = np.array(
        [
            {"clicks": p.cum_clicks, "conversions": p.cum_conversions, "reach": p.cum_reach}[plan.kpi_name]
            for p in plan.trajectory
        ]
    )
    fact_spend = np.zeros(horizon)
    fact_kpi = np.zeros(horizon)
    cum_spend = 0.0
    cum_kpi = 0.0

    for h in range(horizon):
        decision = executor.decide(sim.remaining_budget)
        forecast = executor.forecast(decision.action)
        obs, _, _, _ = sim.step(Action(spend_caps=decision.action))
        events = executor.observe(obs)
        cum_spend += obs.total_spend
        cum_kpi += kpi_of_observation(obs, plan.kpi_name)
        fact_spend[h] = cum_spend
        fact_kpi[h] = cum_kpi
        records.append(
            HourRecord(
                hour=h + 1,
                ml_forecast=forecast,
                ml_signals=dict(executor.ml_signals),
                plan_cum_spend=float(plan_spend[h]),
                plan_cum_kpi=float(plan_kpi[h]),
                fact_cum_spend=cum_spend,
                fact_cum_kpi=cum_kpi,
                fact_by_channel_spend=spend_by_channel(obs),
                fact_by_channel={
                    cid: {
                        "requests": ch.requests,
                        "impressions": ch.impressions,
                        "unique_reach": ch.unique_reach,
                        "clicks": ch.clicks,
                        "conversions": ch.conversions,
                        "spend": ch.spend,
                        "ecpm": ch.ecpm,
                        "fraud_share": ch.fraud_share,
                        "verified_impressions": ch.verified_impressions,
                    }
                    for cid, ch in obs.by_channel.items()
                },
                caps=decision.action,
                deduplicated_reach=obs.total_reach,
                status=decision.status,
                events=events,
                tracking_error_spend=decision.tracking_error_spend,
                tracking_error_kpi=decision.tracking_error_kpi,
                reserve_rub=getattr(executor, "reserve_rub", 0.0),
            )
        )
        if config.stop_at_first_event and events:
            break

    n = len(records)
    shock_hours = sorted({e.start_hour for e in config.injected})
    return RunSummary(
        strategy=config.strategy,
        ml=config.ml,
        ml_model_id=ml_bundle.model_id if config.ml.enabled else None,
        scenario_id=config.scenario_id,
        world_seed=config.seeds.world_seed,
        noise_seed=config.seeds.noise_seed,
        kpi_name=plan.kpi_name,
        promised_spend=float(plan_spend[-1]),
        promised_kpi=float(plan_kpi[-1]),
        actual_spend=cum_spend,
        actual_kpi=cum_kpi,
        mape_spend=mape(plan_spend[:n], fact_spend[:n]),
        mape_kpi=mape(plan_kpi[:n], fact_kpi[:n]),
        wape_spend=wape(plan_spend[:n], fact_spend[:n]),
        wape_kpi=wape(plan_kpi[:n], fact_kpi[:n]),
        final_deviation_spend=final_deviation(plan_spend[:n], fact_spend[:n]),
        final_deviation_kpi=final_deviation(plan_kpi[:n], fact_kpi[:n]),
        unsmoothness=unsmoothness(plan_spend[:n], fact_spend[:n]),
        lambda_cv=coefficient_of_variation(executor.lambdas),
        shock_hours=shock_hours,
        detection_hours=dict(executor.detection_hours),
        human_requests=executor.human_requests,
        proposals=list(executor.proposals),
        hours=records,
        runtime_seconds=time.perf_counter() - started,
    )
