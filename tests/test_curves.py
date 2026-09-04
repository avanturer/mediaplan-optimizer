"""Кривые из ретро-истории: вогнутость, потолок, сглаживание ставок."""

from brain.curves import build_curve


def test_curves_are_concave_and_monotone(curves):
    for cid, curve in curves.items():
        xs = [p.daily_spend for p in curve.points]
        ys = [p.impressions for p in curve.points]
        assert xs == sorted(xs), cid
        assert ys == sorted(ys), cid
        slopes = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
        for a, b in zip(slopes, slopes[1:], strict=False):
            assert b <= a + 1e-9, f"{cid}: предельная отдача выросла с бюджетом"


def test_curve_saturates_where_cap_stops_binding(curves, history):
    """Последняя точка равна потолку: выше неё лимит перестаёт связывать расход."""
    for curve in curves.values():
        assert curve.max_daily_spend > 0
        assert curve.impressions_at(curve.max_daily_spend * 5) == curve.impressions_at(curve.max_daily_spend)
        assert curve.effective_spend(curve.max_daily_spend * 5) == curve.max_daily_spend
    # лестница уровней должна содержать уровни выше потолка, иначе потолок неизвестен
    for cid, curve in curves.items():
        caps = {
            round(sum(a.spend_caps[cid] for a in ep.actions) / max(ep.horizon_hours // 24, 1), 2)
            for ep in history.episodes
        }
        assert max(caps) > curve.max_daily_spend * 0.97, cid


def test_rates_are_prior_smoothed(catalog, history):
    """Оценка CTR остаётся внутри диапазона каталога, расширенного на разумный запас."""
    for channel in catalog.channels:
        curve = build_curve(history, channel)
        lo, hi = channel.expected_ctr_range
        assert lo * 0.5 <= curve.ctr <= hi * 1.5, channel.channel_id
        lo, hi = channel.expected_cvr_range
        assert lo * 0.5 <= curve.cvr <= hi * 1.5, channel.channel_id


def test_hourly_profile_normalised_per_day(curves):
    for curve in curves.values():
        for day in range(7):
            assert abs(curve.hourly_profile[day * 24 : (day + 1) * 24].sum() - 1.0) < 1e-6
