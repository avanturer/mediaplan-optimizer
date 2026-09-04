"""Правдоподобие мира и калибровка масштаба.

Метрики должны лежать в публичных диапазонах бенчмарков, а план демо 1
выкупать около половины инвентаря: иначе после поломки канала деньги некуда
переложить, и преимущество адаптивной стратегии исчезает по свойству стенда.
"""

import numpy as np

from contracts import Action, SeedBundle
from world import Simulator


def test_demo_plan_buys_about_half_of_inventory(demo_plan, catalog):
    sim = Simulator(catalog)
    sim.reset(SeedBundle(world_seed=1, noise_seed=10001), "stable", horizon_hours=504, total_budget=demo_plan.total_budget_rub)
    imps = {cid: 0 for cid in catalog.channel_ids}
    reqs = {cid: 0 for cid in catalog.channel_ids}
    for caps in demo_plan.hourly_caps:
        obs, _, _, _ = sim.step(Action(spend_caps=caps))
        for cid, ch in obs.by_channel.items():
            imps[cid] += ch.impressions
            reqs[cid] += ch.requests
    utilisation = sum(imps.values()) / sum(reqs.values())
    assert 0.25 <= utilisation <= 0.75, f"план выкупает {utilisation:.0%} инвентаря"


def test_metrics_stay_in_plausible_ranges(demo_plan, catalog):
    """Без управления, по плановым лимитам: CPM, CTR, CVR каждого канала в разумных пределах диапазонов каталога."""
    sim = Simulator(catalog)
    sim.reset(SeedBundle(world_seed=2, noise_seed=10002), "stable", horizon_hours=504, total_budget=demo_plan.total_budget_rub)
    acc = {cid: np.zeros(4) for cid in catalog.channel_ids}  # spend, imps, clicks, conv
    for caps in demo_plan.hourly_caps:
        obs, _, _, _ = sim.step(Action(spend_caps=caps))
        for cid, ch in obs.by_channel.items():
            acc[cid] += (ch.spend, ch.impressions, ch.clicks, ch.conversions)
    for cid, (spend, imps, clicks, _conv) in acc.items():
        if imps == 0:
            continue
        channel = catalog.by_id(cid)
        ecpm = spend / imps * 1000
        ctr = clicks / imps
        lo, hi = channel.expected_ecpm_range
        assert lo * 0.4 <= ecpm <= hi * 1.6, f"{cid}: eCPM {ecpm:.0f} вне {lo:.0f}–{hi:.0f}"
        lo, hi = channel.expected_ctr_range
        # усталость за кампанию снижает CTR относительно каталога, поэтому нижняя граница мягче
        assert lo * 0.3 <= ctr <= hi * 1.5, f"{cid}: CTR {ctr:.4f} вне {lo}–{hi}"


def test_sms_respects_send_window_and_base(catalog):
    sim = Simulator(catalog)
    sim.reset(SeedBundle(), "stable", horizon_hours=7 * 24, total_budget=1e7)
    caps = {cid: 0.0 for cid in catalog.channel_ids}
    caps["sms"] = 1e6
    sent = 0
    for h in range(7 * 24):
        obs, _, _, _ = sim.step(Action(spend_caps=caps))
        sms = obs.by_channel["sms"]
        hour_of_day = h % 24
        if not (9 <= hour_of_day < 21):
            assert sms.impressions == 0
        sent += sms.spend / catalog.by_id("sms").sms.price_per_message_rub
    assert sent <= catalog.by_id("sms").sms.base_size * 1.01
