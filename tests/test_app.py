"""Кабинет: три демонстрации кейса через API и правила, важные для показа.

Проверяем не математику (она в тестах brain/world), а контракт кабинета:
что демо 1 отдаёт всё, что требует кейс; что демо 2 отказывает с тремя ходами;
что прогон возможен только для утверждённого плана; что любая ошибка приходит
по-русски как JSON, а не как 500.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

DEMO1 = {
    "mode": "A",
    "preset": "all",
    "budget_rub": 1_200_000,
    "horizon_days": 21,
    "objective": "max_conversions",
    "automation_limit_rub": 50_000,  # с новым исполнителем переносы первого дня ниже 100 000, карточка ждёт решения при 50 000
}
DEMO2 = {"mode": "B", "preset": "narrow", "target_kpi": "clicks", "target_value": 50_000, "horizon_days": 14}
SIX_METRICS = ("ctr", "cvr", "cpm_rub", "cpc_rub", "cpa_rub", "vtr")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # контекст запускает lifespan: каталог, ретро, кривые
        yield c


def _plan(client: TestClient, body: dict) -> dict:
    r = client.post("/api/plan", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _approved_plan(client: TestClient, body: dict) -> dict:
    plan = _plan(client, body)
    r = client.post(f"/api/plan/{plan['plan_id']}/approve")
    assert r.status_code == 200, r.text
    return plan


def test_demo1_plan_has_everything_case_asks(client):
    plan = _plan(client, DEMO1)
    assert plan["infeasibility"] is None
    assert len(plan["allocations"]) == 8
    assert abs(plan["total_budget_rub"] - 1_200_000) < 1.0
    for a in plan["allocations"]:
        for key in SIX_METRICS:
            assert key in a, f"в строке плана нет {key}"
        assert a["display_name"] and not a["display_name"].startswith(a["channel_id"])
    assert any(a["marginal_cost_per_1000_kpi_rub"] for a in plan["allocations"]), "цена следующей тысячи не посчитана"
    assert len(plan["calendar"]) == 8 * 21
    assert len(plan["trajectory"]) == 21 * 24
    f = plan["forecast"]
    assert f["p10"] < f["p50"] < f["p90"]
    assert plan["trajectory"][-1]["cum_conversions"] == pytest.approx(plan["total_kpi"])


def test_demo2_refuses_with_three_moves_cheapest_first(client):
    plan = _plan(client, DEMO2)
    inf = plan["infeasibility"]
    assert inf is not None
    assert inf["max_achievable"] < 50_000
    assert inf["binding_title"] and inf["binding_title"] != inf["binding_constraint"]
    budgets = [s["expected_budget_rub"] for s in inf["suggestions"]]
    assert len(budgets) == 3
    assert budgets == sorted(budgets), "ходы должны идти от дешёвого к дорогому"
    fields = {s["changed_field"] for s in inf["suggestions"]}
    assert fields == {"horizon_days", "target_value", "channel_ids"}


def test_infeasible_plan_cannot_be_approved_or_run(client):
    plan = _plan(client, DEMO2)
    assert client.post(f"/api/plan/{plan['plan_id']}/approve").status_code == 409
    r = client.post("/api/run", json={"plan_id": plan["plan_id"]})
    assert r.status_code == 409
    assert "недостижим" in r.json()["detail"]  # достижимость проверяется раньше утверждения


def test_run_requires_approval(client):
    plan = _plan(client, {**DEMO1, "horizon_days": 20})  # отдельный план, чтобы не зависеть от порядка тестов
    r = client.post("/api/run", json={"plan_id": plan["plan_id"]})
    assert r.status_code == 409
    assert "Утвердить план" in r.json()["detail"]


def test_approve_is_idempotent_and_run_is_reproducible(client):
    plan = _plan(client, DEMO1)
    pid = plan["plan_id"]
    assert client.post(f"/api/plan/{pid}/approve").json()["approved_version"] == 1
    assert client.post(f"/api/plan/{pid}/approve").json()["approved_version"] == 1

    body = {"plan_id": pid, "scenario_id": "channel_pause", "world_seed": 1}
    r = client.post("/api/run", json=body)
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["plan_id"] == pid and run["approved_version"] == 1
    assert run["seeds"] == {"catalog_seed": 0, "world_seed": 1, "noise_seed": 10_001}
    assert len(run["main"]["hours"]) == len(run["frozen"]["hours"]) == 21 * 24
    hour = run["main"]["hours"][100]
    for key in ("requests", "impressions", "unique_reach", "clicks", "conversions", "spend", "ecpm"):
        assert key in hour["by_channel"]["programmatic"], f"в почасовом ряду нет {key}"
    assert hour["status"] in ("ok", "watch", "fire")
    v = run["verdict"]
    assert v["threshold"] == 0.2 and "within_threshold" in v and "frozen_within_threshold" in v
    # тот же запрос → та же кампания: воспроизводимость по зёрнам
    again = client.post("/api/run", json=body).json()
    assert again["verdict"]["actual_kpi"] == run["verdict"]["actual_kpi"]


def test_decide_rejects_hour_without_card(client):
    plan = _approved_plan(client, DEMO1)
    run = client.post("/api/run", json={"plan_id": plan["plan_id"]}).json()
    r = client.post(f"/api/run/{run['run_id']}/decide", json={"hour": 5, "decision": "approve"})
    assert r.status_code == 422
    assert "карточки" in r.json()["detail"]


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"mode": "A", "budget_rub": 1_000_000, "horizon_days": 0}, "срок"),
        ({"mode": "A", "budget_rub": 0}, "бюджет"),
        ({"mode": "B", "target_kpi": "clicks"}, "целевой объём"),
        ({"mode": "A", "preset": "nope", "budget_rub": 1_000_000}, "пресет"),
        ({"mode": "A", "budget_rub": 100_000, "locked": {"sms": 500_000}}, "больше бюджета"),
        ({"mode": "A", "budget_rub": 1_000_000, "channel_ids": []}, "пуст"),
        ({"mode": "A", "budget_rub": 1_000_000, "horizon_days": 90}, "не больше 30"),
    ],
)
def test_bad_brief_is_russian_422(client, body, fragment):
    r = client.post("/api/plan", json=body)
    assert r.status_code == 422, r.text
    assert fragment in r.json()["detail"]


def test_bad_run_inputs_are_russian_422(client):
    plan = _approved_plan(client, DEMO1)
    pid = plan["plan_id"]
    cases = [
        ({"plan_id": pid, "strategy": "lstm"}, "стратегия"),
        ({"plan_id": pid, "scenario_id": "nope"}, "сценарий"),
        ({"plan_id": pid, "shock": {"channel_id": "sms", "start_hour": 600}}, "за пределами кампании"),
        ({"plan_id": pid, "shock": {"channel_id": "sms", "multiplier": -1}}, "сила шока"),
        ({"plan_id": pid, "shock": {"channel_id": "sms", "parameter": "foo"}}, "параметр шока"),
    ]
    for body, fragment in cases:
        r = client.post("/api/run", json=body)
        assert r.status_code == 422, (body, r.text)
        assert fragment in r.json()["detail"], (body, r.json())


def test_shock_channel_must_belong_to_plan(client):
    plan = _approved_plan(client, {**DEMO1, "preset": "performance"})
    r = client.post("/api/run", json={"plan_id": plan["plan_id"], "shock": {"channel_id": "sms"}})
    assert r.status_code == 422
    assert "нет в плане" in r.json()["detail"]


def test_unknown_plan_is_404_not_500(client):
    r = client.post("/api/run", json={"plan_id": "deadbeef"})
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_custom_shocks_list_and_legacy_single_field(client):
    plan = _approved_plan(client, DEMO1)
    pid = plan["plan_id"]
    two = [
        {"channel_id": "sms", "parameter": "cvr", "multiplier": 0.5, "start_hour": 0},
        {"channel_id": "programmatic", "parameter": "ecpm", "multiplier": 2, "start_hour": 240},
    ]
    r = client.post("/api/run", json={"plan_id": pid, "shocks": two})
    assert r.status_code == 200, r.text
    run = r.json()
    assert sorted(run["main"]["shock_hours"]) == [0, 240]
    assert [s["channel_id"] for s in run["custom_shocks"]] == ["sms", "programmatic"]
    legacy = client.post("/api/run", json={"plan_id": pid, "shock": {"channel_id": "sms", "start_hour": 24}}).json()
    assert [s["channel_id"] for s in legacy["custom_shocks"]] == ["sms"]
    r = client.post("/api/run", json={"plan_id": pid, "shocks": [{"channel_id": "sms"}] * 4})
    assert r.status_code == 422
    assert "не больше 3" in r.json()["detail"]


def test_empty_plan_cannot_be_approved(client):
    plan = _plan(client, {**DEMO1, "max_cpa_rub": 10})  # потолок ниже любого канала: все бюджеты нулевые
    assert plan["is_empty"] is True and plan["total_budget_rub"] == 0
    r = client.post(f"/api/plan/{plan['plan_id']}/approve")
    assert r.status_code == 409
    assert "пуст" in r.json()["detail"]


def test_suggestion_beyond_cabinet_horizon_is_marked_not_applicable(client):
    plan = _plan(client, {"mode": "B", "preset": "narrow", "target_kpi": "clicks", "target_value": 90_000, "horizon_days": 21})
    sug = {s["changed_field"]: s for s in plan["infeasibility"]["suggestions"]}
    horizon = sug["horizon_days"]
    assert horizon["suggested_value"] > 30
    assert horizon["applicable"] is False and "30" in horizon["why_not"]
    assert all(s["applicable"] for f, s in sug.items() if f != "horizon_days")


def test_infeasible_plan_run_explains_infeasibility_not_approval(client):
    plan = _plan(client, DEMO2)
    r = client.post("/api/run", json={"plan_id": plan["plan_id"]})
    assert r.status_code == 409
    assert "недостижим" in r.json()["detail"]


def test_decide_approve_reruns_and_decline_after_approve_is_rejected(client):
    plan = _approved_plan(client, DEMO1)
    run = client.post("/api/run", json={"plan_id": plan["plan_id"], "scenario_id": "channel_pause"}).json()
    pending = [p["hour"] for p in run["main"]["proposals"] if p["applied_by"] == "pending"]
    assert pending, "в демо-плане с лимитом 50 000 ₽ должна быть карточка, ждущая решения"
    r = client.post(f"/api/run/{run['run_id']}/decide", json={"hour": pending[0], "decision": "approve"})
    assert r.status_code == 200, r.text
    after = r.json()
    assert after["run_id"] != run["run_id"] and after["effect"]["hour"] == pending[0]
    assert any(p["hour"] == pending[0] and p["applied_by"] == "human" for p in after["main"]["proposals"])
    r = client.post(f"/api/run/{after['run_id']}/decide", json={"hour": pending[0], "decision": "decline"})
    assert r.status_code == 422
    # откат своего одобрения: карточка снова ждёт решения, итог возвращается к исходному
    r = client.post(f"/api/run/{after['run_id']}/decide", json={"hour": pending[0], "decision": "undo"})
    assert r.status_code == 200, r.text
    undone = r.json()
    assert undone["effect"]["hour"] == pending[0]
    assert any(p["hour"] == pending[0] and p["applied_by"] == "pending" for p in undone["main"]["proposals"])
    assert undone["verdict"]["actual_kpi"] == run["verdict"]["actual_kpi"]
    r = client.post(f"/api/run/{undone['run_id']}/decide", json={"hour": pending[0], "decision": "undo"})
    assert r.status_code == 422  # отменять уже нечего


def test_static_strategy_twin_equals_main(client):
    plan = _approved_plan(client, DEMO1)
    run = client.post("/api/run", json={"plan_id": plan["plan_id"], "strategy": "static"}).json()
    assert run["frozen"]["actual_kpi"] == run["verdict"]["actual_kpi"]
    assert len(run["frozen"]["hours"]) == len(run["main"]["hours"])
