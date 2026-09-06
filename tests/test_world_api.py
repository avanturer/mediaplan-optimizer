from fastapi.testclient import TestClient

from app.main import app


def test_targeted_brief_and_competitor_run_through_api():
    with TestClient(app) as client:
        target = {"geo": ["capital"], "age_groups": ["25_34"]}
        response = client.post("/api/plan", json={"mode": "A", "budget_rub": 100_000,
            "horizon_days": 14, "targeting": target})
        assert response.status_code == 200, response.text
        media_plan = response.json()
        assert media_plan["brief"]["targeting"]["geo"] == ["capital"]
        response = client.post("/api/run", json={"plan_id": media_plan["plan_id"], "strategy": "static",
            "scenario_id": "fraud_surge", "world_settings": {"competitors": [{"competitor_id": "rival",
                "channel_advantages": {"programmatic": 1}}]}})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result["main"]["hours"]) == 336
        assert result["main"] == result["frozen"]
        row = result["main"]["hours"][250]
        assert "fraud_share" in row["by_channel"]["programmatic"]
        assert "deduplicated_reach" in row
        response = client.post("/api/run", json={"plan_id": media_plan["plan_id"],
            "shock": {"channel_id": "sms", "parameter": "fraud", "multiplier": 10}})
        assert response.status_code == 422
