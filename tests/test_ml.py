"""ML uses public past observations; switches and paired controls are independent."""

import json

import numpy as np
import pytest

from brain.executor import make_executor
from brain.ml import MLBundle, QualityWindow
from brain.planner import plan
from contracts import Action, Brief, SeedBundle
from contracts.ml import MLConfig
from harness.ml_training import train_ml_bundle
from harness.runner import RunConfig, run_campaign
from world import Simulator


@pytest.fixture(scope="module")
def bundle(catalog, curves, history):
    return train_ml_bundle(catalog, curves, history)


def test_artifact_roundtrip(bundle):
    restored = MLBundle.from_dict(json.loads(json.dumps(bundle.to_dict())))
    assert restored.model_id == bundle.model_id
    assert restored.quality.threshold == bundle.quality.threshold
    assert restored.training_summary["positive_training_rows"] > 0


def test_response_and_reach_constraints(bundle):
    for curve in bundle.curves.values():
        for p in curve.points:
            assert 0 <= p.conversions <= p.clicks <= p.impressions
        for key in ("impressions", "clicks", "conversions"):
            assert np.all(np.diff([getattr(p, key) for p in curve.points]) >= -1e-8)
    model = bundle.reach
    r = {cid: pool*.3 for cid, pool in model.pools.items()}
    assert max(r.values()) <= model.predict(r) < sum(r.values())
    for cid in r:
        assert model.incremental({cid: 100}, r) >= 0
    assert model.predict({}) == 0
    assert model.incremental({}, r) == 0


@pytest.mark.parametrize("strategy", ["static", "adaptive"])
def test_ml_off_is_identical(strategy, bundle, demo_plan, catalog, curves):
    base = run_campaign(demo_plan, catalog, curves, RunConfig(strategy=strategy))
    off = run_campaign(demo_plan, catalog, curves, RunConfig(strategy=strategy, ml=MLConfig()), ml_bundle=bundle)
    assert base.hours == off.hours
    assert not any(h.ml_forecast or h.ml_signals for h in off.hours)


@pytest.mark.parametrize("flag", ["anomaly_detection", "response_curves", "reach_correction"])
def test_switches_and_forecast_order(flag, bundle, demo_plan, catalog, curves):
    config = MLConfig(**{flag: True})
    ex = make_executor("adaptive", demo_plan, catalog, curves, demo_plan.total_budget_rub,
                       ml_config=config, ml_bundle=bundle, auto_apply_above_limit=False)
    assert (ex.curves is bundle.curves) == config.response_curves
    assert (ex.reach_model is not None) == config.reach_correction
    assert (ex.quality_monitor is not None) == config.anomaly_detection
    sim = Simulator(catalog)
    sim.reset(SeedBundle(world_seed=987), "fraud_surge", total_budget=demo_plan.total_budget_rub)
    for _ in range(30):
        decision = ex.decide(sim.remaining_budget)
        forecast = ex.forecast(decision.action)
        assert forecast.generated_at_hour == ex.hour
        obs, _, _, _ = sim.step(Action(spend_caps=decision.action))
        assert forecast.forecast_for_hour == obs.hour
        assert forecast.predicted_reach <= forecast.additive_predicted_reach + 1e-6
        ex.observe(obs)
    zero = ex.forecast(dict.fromkeys(ex.channel_ids, 0))
    assert zero.predicted_kpi == zero.predicted_spend == 0


def test_reach_plan_and_missing_model(bundle, catalog, curves):
    brief = Brief(budget_rub=100000, horizon_days=14, objective="max_reach", channel_ids=catalog.channel_ids,
                  ml=MLConfig(reach_correction=True))
    with pytest.raises(ValueError, match="артефакт"):
        plan(brief, catalog, curves)
    result = plan(brief, catalog, curves, bundle)
    reached = {a.channel_id: a.unique_reach for a in result.allocations}
    assert result.total_kpi == pytest.approx(bundle.reach.predict(reached))
    assert result.forecast.p50 == result.total_kpi
    assert np.all(np.diff([p.cum_reach for p in result.trajectory]) >= 0)


def test_quality_no_volume(catalog):
    sim = Simulator(catalog)
    sim.reset(SeedBundle(), "stable")
    window = QualityWindow()
    for _ in range(25):
        obs, _, _, _ = sim.step(Action(spend_caps=dict.fromkeys(catalog.channel_ids, 0)))
        assert window.update(obs.by_channel["programmatic"], 0) is None


def test_model_replay_and_human_limit(bundle, catalog, curves):
    brief = Brief(budget_rub=100000, horizon_days=14, channel_ids=catalog.channel_ids,
                  automation_limit_rub=0)
    media_plan = plan(brief, catalog, curves)
    config = RunConfig(ml=MLConfig(anomaly_detection=True, response_curves=True, reach_correction=True),
                       auto_apply_above_limit=False, scenario_id="fraud_surge")
    run = run_campaign(media_plan, catalog, curves, config, ml_bundle=bundle)
    restored = MLBundle.from_dict(json.loads(json.dumps(bundle.to_dict())))
    replay = run_campaign(media_plan, catalog, curves, config, ml_bundle=restored)
    assert run.hours == replay.hours
    assert run.proposals
    assert all(p.applied_by == "pending" for p in run.proposals)
    assert run.actual_spend <= media_plan.total_budget_rub+1e-6
