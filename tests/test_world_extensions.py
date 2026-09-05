"""Поведенческие проверки сегментов, общих пулов, фрода, квот и конкурентов."""

import json
import time

import numpy as np
import pytest

from brain.executor.controller import kpi_of_observation
from brain.planner import plan
from contracts import Action, SeedBundle, ShockEvent, ShockParameter
from contracts.simulation import Observation, Scenario
from contracts.targeting import AudienceTargeting
from world import Simulator, build_catalog
from world.audience import Audience
from world.competition import competition_tapes
from world.settings import Competitor, WorldSettings
from world.targeting import catalog_for_targeting


def run_sim(settings=None, scenario="stable", hours=72, cap=1000, seed=3, targeting=None):
    sim = Simulator(settings=settings)
    sim.reset(SeedBundle(world_seed=seed, noise_seed=seed + 100), scenario,
              horizon_hours=hours, total_budget=1e12, targeting=targeting)
    action = Action(spend_caps=dict.fromkeys(sim.catalog.channel_ids, cap))
    rows = [sim.step(action)[0] for _ in range(hours)]
    return sim, rows


def shock(parameter, channel, start=24, duration=24, multiplier=10, recovery="linear"):
    return Scenario(scenario_id="test", events=[ShockEvent(
        start_hour=start, duration_hours=duration, target_channels=[channel],
        parameter=parameter, multiplier=multiplier, recovery=recovery)])


def test_targeting_changes_public_catalog_capacity_price_and_delivery():
    target = AudienceTargeting(age_groups=["25_34"], genders=["female"], geo=["capital"])
    catalog = build_catalog()
    narrow = catalog_for_targeting(catalog, target)
    assert narrow.catalog_id != catalog.catalog_id
    assert catalog.targeting == AudienceTargeting()
    assert narrow == catalog_for_targeting(catalog, target)
    for ch in narrow.channels:
        assert ch.capacity_mid < catalog.by_id(ch.channel_id).capacity_mid
        assert ch.ecpm_mid > catalog.by_id(ch.channel_id).ecpm_mid
    _, broad_rows = run_sim(hours=24, cap=1e7)
    _, narrow_rows = run_sim(hours=24, cap=1e7, targeting=target)
    assert sum(o.total_reach for o in narrow_rows) < sum(o.total_reach for o in broad_rows)
    assert sum(o.total_impressions for o in narrow_rows) < sum(o.total_impressions for o in broad_rows)
    assert sum(o.by_channel["social_1"].ecpm for o in narrow_rows) > sum(o.by_channel["social_1"].ecpm for o in broad_rows)


def test_invalid_targeting_and_mismatched_plan_rejected(demo_brief, catalog, curves):
    with pytest.raises(ValueError):
        AudienceTargeting(geo=["unknown"])
    with pytest.raises(ValueError):
        AudienceTargeting(age_min=21)
    target = AudienceTargeting(genders=["female", "female"])
    assert target.genders == ["female"]
    with pytest.raises(ValueError, match="таргетинг"):
        plan(demo_brief.model_copy(update={"targeting": target}), catalog, curves)


def test_pair_matrix_has_exact_pool_sizes_and_analytic_union():
    matrix = {"a": {"a": 1, "b": 0.5}, "b": {"a": 0.5, "b": 1}}
    audience = Audience({"a": 1000, "b": 2000}, WorldSettings(overlap_matrix=matrix))
    assert sum(c.size for c in audience.cohorts) == 2500
    observed = audience.step({"a": 1000, "b": 2000})
    expected = 2000 * (1 - np.exp(-1)) + 500 * (1 - np.exp(-2))
    assert observed == round(expected)
    # Уже достигнутые через A люди дают меньше нового охвата при подключении B.
    a_then_b = Audience({"a": 1000, "b": 2000}, WorldSettings(overlap_matrix=matrix))
    a_then_b.step({"a": 10_000})
    b_increment = a_then_b.step({"b": 2000})
    fresh = Audience({"a": 1000, "b": 2000}, WorldSettings(overlap_matrix=matrix))
    assert b_increment < fresh.step({"b": 2000})


@pytest.mark.parametrize("matrix", [
    {"a": {"a": 1, "b": 0.1}, "b": {"a": 0.2, "b": 1}},
    {"a": {"a": 1, "b": float("nan")}, "b": {"a": 0.2, "b": 1}},
    {"a": {"a": 1, "b": -0.1}, "b": {"a": -0.1, "b": 1}},
    {"a": {"a": 0, "b": 0.1}, "b": {"a": 0.1, "b": 1}},
    {"a": {"a": 1}},
])
def test_invalid_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        Audience({"a": 1000, "b": 2000}, WorldSettings(overlap_matrix=matrix))


def test_infeasible_pair_only_matrix_rejected():
    with pytest.raises(ValueError, match="превышает"):
        Audience({"a": 1000, "b": 1000, "c": 1000}, WorldSettings(default_overlap=0.6))


def test_overlap_reduces_campaign_reach_without_changing_channel_delivery():
    _, independent = run_sim(WorldSettings(default_overlap=0), hours=168)
    _, shared = run_sim(WorldSettings(default_overlap=0.1), hours=168)
    assert sum(o.total_reach for o in shared) < sum(o.total_reach for o in independent)
    for a, b in zip(independent, shared, strict=True):
        assert a.by_channel == b.by_channel
        assert kpi_of_observation(b, "reach") == b.deduplicated_reach


def test_fraud_increases_ctr_reduces_conversions_and_reports_estimate():
    scenario = shock(ShockParameter.FRAUD, "programmatic", start=0, duration=48, multiplier=15)
    totals = np.zeros((2, 5))
    for seed in range(8):
        for i, scene in enumerate(("stable", scenario)):
            _, rows = run_sim(scenario=scene, hours=48, seed=seed)
            for row in rows:
                ch = row.by_channel["programmatic"]
                totals[i] += [ch.impressions, ch.clicks, ch.conversions, ch.fraud_share * ch.impressions, ch.spend]
                assert ch.verified_impressions <= ch.impressions
    normal, fraud = totals
    assert fraud[1] / fraud[0] > normal[1] / normal[0]
    assert fraud[2] < normal[2]
    assert normal[3] / normal[0] == pytest.approx(0.03, abs=0.005)
    assert fraud[3] / fraud[0] > 0.05
    assert fraud[4] == normal[4]


def test_fraud_timing_recovery_and_no_effect_on_other_channels():
    scenario = shock(ShockParameter.FRAUD, "programmatic", duration=12)
    _, base = run_sim(hours=72)
    _, rows = run_sim(scenario=scenario, hours=72)
    for h in range(72):
        for cid in rows[h].by_channel:
            if cid != "programmatic" or h < 24:
                assert rows[h].by_channel[cid] == base[h].by_channel[cid]
        if h >= 48:
            # После recovery доля ботов и измеритель возвращаются на ту же ленту.
            assert rows[h].by_channel["programmatic"].fraud_share == base[h].by_channel["programmatic"].fraud_share
    event = scenario.events[0]
    assert event.factor_at(23) is None
    assert event.factor_at(24) == 10
    assert event.factor_at(42) == 5.5
    assert event.factor_at(48) is None


def test_all_bot_delivery_has_no_conversions_or_human_reach():
    scenario = shock(ShockParameter.FRAUD, "programmatic", start=0, multiplier=100)
    _, rows = run_sim(scenario=scenario, hours=24)
    for row in rows:
        ch = row.by_channel["programmatic"]
        assert ch.conversions == ch.unique_reach == 0


def test_weekly_sms_quota_reset_and_recovery():
    scenario = shock(ShockParameter.SMS_WEEKLY_LIMIT, "sms", start=0, duration=336, multiplier=0.25, recovery="none")
    sim, rows = run_sim(scenario=scenario, hours=504, cap=1e7)
    sms = sim.catalog.by_id("sms").sms
    weekly = [sum(o.by_channel["sms"].spend / sms.price_per_message_rub for o in rows[start:start+168])
              for start in (0, 168, 336)]
    assert 0 < weekly[0] <= sms.base_size * 0.25 + 1e-6
    assert weekly[1] == pytest.approx(weekly[0])
    assert weekly[2] > 3 * weekly[1]


def test_sms_midweek_limit_counts_already_sent_messages():
    scenario = shock(ShockParameter.SMS_WEEKLY_LIMIT, "sms", start=48, duration=168, multiplier=0.10, recovery="none")
    _, rows = run_sim(scenario=scenario, hours=180, cap=1e7)
    assert sum(o.by_channel["sms"].spend for o in rows[:48]) > 0
    assert sum(o.by_channel["sms"].spend for o in rows[48:168]) == 0
    assert sum(o.by_channel["sms"].spend for o in rows[168:]) > 0


def test_competitors_are_optional_bounded_paired_and_channel_specific():
    settings = WorldSettings(competitors=[Competitor(competitor_id="rival", strength=0.8,
        channel_advantages={"social_1": 2}, start_hour=24, end_hour=48)])
    _, base = run_sim(hours=72, cap=1e7)
    _, rival = run_sim(settings, hours=72, cap=1e7)
    for h, (a, b) in enumerate(zip(base, rival, strict=True)):
        for cid in a.by_channel:
            if cid != "social_1" or h < 24:
                assert a.by_channel[cid] == b.by_channel[cid]
        if 24 <= h < 48:
            assert b.by_channel["social_1"].ecpm > a.by_channel["social_1"].ecpm
        if h >= 48:
            assert a.by_channel["social_1"].requests == b.by_channel["social_1"].requests
    assert sum(o.by_channel["social_1"].impressions for o in rival) < sum(o.by_channel["social_1"].impressions for o in base)
    _, low = run_sim(settings, hours=72, cap=100)
    for a, b in zip(low, rival, strict=True):
        assert a.by_channel["social_1"].requests == b.by_channel["social_1"].requests
    pressure = competition_tapes(settings, 103, ["social_1"], 72)["social_1"]
    assert np.all((pressure >= 0) & (pressure <= 2))
    assert np.std(pressure[24:48]) > 0


def test_extended_manifest_full_21_day_replay_and_no_leakage():
    settings = WorldSettings(competitors=[Competitor(competitor_id="rival", channel_advantages={"social_1": 1})])
    start = time.perf_counter()
    sim, rows = run_sim(settings, scenario="fraud_surge", hours=504, targeting=AudienceTargeting(geo=["large_cities"]))
    assert time.perf_counter() - start < 5
    manifest = json.loads(json.dumps(sim.export_manifest()))
    replay = Simulator.from_manifest(manifest)
    action = Action(spend_caps=dict.fromkeys(sim.catalog.channel_ids, 1000))
    repeated = []
    for _ in range(504):
        obs, metrics, done, info = replay.step(action)
        repeated.append(obs)
        payload = json.dumps([obs.model_dump(), metrics.model_dump(), info.model_dump()])
        for forbidden in ("competitor", "noise_seed", "world_seed", "overlap_matrix", "fraud_baseline", "base_ctr", "unique_pool"):
            assert forbidden not in payload
        assert 0 <= obs.total_reach <= obs.total_impressions
        assert metrics.cumulative_spend <= 1e12
    assert repeated == rows
    assert done
    with pytest.raises(RuntimeError):
        replay.step(action)
    assert Observation.model_validate_json(rows[-1].model_dump_json()) == rows[-1]


def test_zero_caps_with_all_extensions_and_invalid_shocks():
    sim, rows = run_sim(scenario="fraud_surge", hours=504, cap=0)
    assert all(o.total_spend == o.total_impressions == o.total_reach == 0 for o in rows)
    sim.reset(SeedBundle())
    with pytest.raises(ValueError, match="только"):
        sim.inject_shock(shock(ShockParameter.FRAUD, "sms").events[0])
    with pytest.raises(ValueError, match="только"):
        sim.inject_shock(shock(ShockParameter.SMS_WEEKLY_LIMIT, "social_1", multiplier=0.5).events[0])


def test_manifest_changes_with_settings_scenario_and_injected_shock():
    sim = Simulator()
    _, initial = sim.reset(SeedBundle())
    event = shock(ShockParameter.FRAUD, "programmatic").events[0]
    sim.inject_shock(event)
    manifest = sim.export_manifest()
    replay = Simulator.from_manifest(manifest)
    assert replay.export_manifest() == manifest
    assert sim.step(Action(spend_caps=dict.fromkeys(sim.catalog.channel_ids, 0)))[3].config_hash != initial.config_hash
    manifest["world_fingerprint"] = "changed"
    with pytest.raises(ValueError):
        Simulator.from_manifest(manifest)
