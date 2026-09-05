"""Декларативные сценарии шоков (контракт мира, §10).

Минимальный набор из контракта: стабильный мир, скачок CPM, падение CTR,
падение CVR, исчерпание инвентаря, пауза канала, всплеск спроса, постепенное
восстановление. Оптимизатор расписание не получает; runner знает сценарий
только для организации эксперимента.
"""

from contracts.simulation import Scenario, ShockEvent, ShockParameter

MID_HORIZON_HOUR = 240  # десятые сутки 21-дневной кампании

SCENARIOS: dict[str, Scenario] = {
    "stable": Scenario(scenario_id="stable", events=[]),
    "fraud_surge": Scenario(
        scenario_id="fraud_surge",
        events=[ShockEvent(start_hour=240, duration_hours=48, target_channels=["programmatic"],
                           parameter=ShockParameter.FRAUD, multiplier=10.0, recovery="linear")],
    ),
    "sms_weekly_limit": Scenario(
        scenario_id="sms_weekly_limit",
        events=[ShockEvent(start_hour=168, duration_hours=168, target_channels=["sms"],
                           parameter=ShockParameter.SMS_WEEKLY_LIMIT, multiplier=0.5)],
    ),
    "ctr_drop": Scenario(
        scenario_id="ctr_drop",
        events=[
            ShockEvent(
                start_hour=MID_HORIZON_HOUR,
                duration_hours=None,
                target_channels=["marketplace_1"],
                parameter=ShockParameter.CTR,
                multiplier=0.6,
            )
        ],
    ),
    "cpm_spike": Scenario(
        scenario_id="cpm_spike",
        events=[
            ShockEvent(
                start_hour=MID_HORIZON_HOUR,
                duration_hours=None,
                target_channels=["marketplace_1"],
                parameter=ShockParameter.ECPM,
                multiplier=2.0,
            )
        ],
    ),
    "cpm_spike_recovery": Scenario(
        scenario_id="cpm_spike_recovery",
        events=[
            ShockEvent(
                start_hour=168,
                duration_hours=48,
                target_channels=["marketplace_1"],
                parameter=ShockParameter.ECPM,
                multiplier=1.4,
                recovery="linear",
            )
        ],
    ),
    "cvr_drop": Scenario(
        scenario_id="cvr_drop",
        events=[
            ShockEvent(
                start_hour=MID_HORIZON_HOUR,
                duration_hours=None,
                target_channels=["marketplace_3"],
                parameter=ShockParameter.CVR,
                multiplier=0.6,
            )
        ],
    ),
    "capacity_cut": Scenario(
        scenario_id="capacity_cut",
        events=[
            ShockEvent(
                start_hour=MID_HORIZON_HOUR,
                duration_hours=96,
                target_channels=["sms"],
                parameter=ShockParameter.INVENTORY,
                multiplier=0.5,
            )
        ],
    ),
    "channel_pause": Scenario(
        scenario_id="channel_pause",
        events=[
            ShockEvent(
                start_hour=MID_HORIZON_HOUR,
                duration_hours=72,
                target_channels=["programmatic"],
                parameter=ShockParameter.PAUSE,
                multiplier=1.0,
            )
        ],
    ),
    "demand_surge": Scenario(
        scenario_id="demand_surge",
        events=[
            ShockEvent(
                start_hour=MID_HORIZON_HOUR,
                duration_hours=48,
                target_channels=["social_1", "social_2", "social_3"],
                parameter=ShockParameter.DEMAND,
                multiplier=1.6,
            )
        ],
    ),
}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise KeyError(f"неизвестный сценарий {scenario_id!r}; есть: {sorted(SCENARIOS)}") from exc
