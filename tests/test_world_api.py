"""Расширения мира доходят до кабинета: сегмент, конкуренты, фрод, дедуплицированный охват.

Тест переписан под контракт кабинета после слияния веток: план утверждается перед
прогоном, шоки приходят списком ``shocks``, ответ прогона это ``main`` и ``frozen``.
"""

from fastapi.testclient import TestClient

from app.main import app


def test_targeted_brief_and_competitor_run_through_api():
    with TestClient(app) as client:
        target = {"geo": ["capital"], "age_groups": ["25_34"]}
        response = client.post(
            "/api/plan",
            json={"mode": "A", "budget_rub": 100_000, "horizon_days": 14, "targeting": target},
        )
        assert response.status_code == 200, response.text
        media_plan = response.json()
        assert media_plan["brief"]["targeting"]["geo"] == ["capital"]
        # узкий сегмент дороже и меньше: ёмкость каналов в плане не может остаться широкой
        assert media_plan["total_budget_rub"] <= 100_000 + 1e-6

        plan_id = media_plan["plan_id"]
        assert client.post(f"/api/plan/{plan_id}/approve").status_code == 200
        response = client.post(
            "/api/run",
            json={
                "plan_id": plan_id,
                "strategy": "static",
                "scenario_id": "fraud_surge",
                "world_settings": {"competitors": [{"competitor_id": "rival", "channel_advantages": {"programmatic": 1}}]},
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result["main"]["hours"]) == 336
        row = result["main"]["hours"][250]
        assert "fraud_share" in row["by_channel"]["programmatic"]
        assert "deduplicated_reach" in row


def test_fraud_shock_is_rejected_outside_programmatic():
    with TestClient(app) as client:
        media_plan = client.post(
            "/api/plan", json={"mode": "A", "budget_rub": 100_000, "horizon_days": 14}
        ).json()
        plan_id = media_plan["plan_id"]
        client.post(f"/api/plan/{plan_id}/approve")
        response = client.post(
            "/api/run",
            json={"plan_id": plan_id, "shocks": [{"channel_id": "sms", "parameter": "fraud", "multiplier": 10}]},
        )
        assert response.status_code == 422, response.text
