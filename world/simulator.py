"""Симулятор рынка: reset / step / inject_shock.

Реализует публичный контракт модели мира (docs/world/WORLD_MODEL.md, §8):
один вызов ``step`` разыгрывает один час для всех активных каналов и
возвращает только агрегаты. Скрытое состояние наружу не выходит, ``info``
содержит лишь безопасные идентификаторы и сообщения валидации.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from contracts.catalog import PublicCatalog
from contracts.simulation import (
    API_VERSION,
    Action,
    ChannelObservation,
    Observation,
    Scenario,
    SeedBundle,
    ShockEvent,
    ShockParameter,
    StepInfo,
    StepMetrics,
)
from world.catalog_builder import build_catalog
from world.engine import ChannelState, step_channel
from world.params import HiddenChannelParams, draw_hidden_params
from world.rng import ar1_tape
from world.scenarios import get_scenario

NOISE_EVENTS = ("traffic", "price", "ctr", "cvr")


@dataclass
class _Episode:
    seeds: SeedBundle
    scenario: Scenario
    horizon_hours: int
    total_budget: float
    channel_ids: list[str]
    params: dict[str, HiddenChannelParams]
    tapes: dict[str, dict[str, np.ndarray]]
    states: dict[str, ChannelState] = field(default_factory=dict)
    hour: int = 0
    cumulative_spend: float = 0.0
    cumulative_clicks: int = 0
    cumulative_conversions: int = 0
    injected: list[ShockEvent] = field(default_factory=list)
    terminated: bool = False

    @property
    def episode_id(self) -> str:
        payload = json.dumps(
            {
                "seeds": self.seeds.model_dump(),
                "scenario": self.scenario.scenario_id,
                "horizon": self.horizon_hours,
                "budget": self.total_budget,
                "channels": self.channel_ids,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class Simulator:
    """Мир восьми каналов с шагом в один час."""

    def __init__(self, catalog: PublicCatalog | None = None, catalog_seed: int = 0) -> None:
        self.catalog = catalog or build_catalog(catalog_seed)
        self._episode: _Episode | None = None

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed_bundle: SeedBundle,
        scenario_id: str | Scenario = "stable",
        horizon_hours: int = 21 * 24,
        total_budget: float = 1_200_000.0,
        channel_ids: list[str] | None = None,
    ) -> tuple[Observation, StepInfo]:
        scenario = scenario_id if isinstance(scenario_id, Scenario) else get_scenario(scenario_id)
        channel_ids = list(channel_ids or self.catalog.channel_ids)
        unknown = set(channel_ids) - set(self.catalog.channel_ids)
        if unknown:
            raise ValueError(f"неизвестные каналы: {sorted(unknown)}")
        params = draw_hidden_params(self.catalog, seed_bundle.world_seed)
        tapes = {
            cid: {
                event: ar1_tape(
                    seed_bundle.noise_seed,
                    cid,
                    event,
                    horizon_hours,
                    params[cid].noise_rho,
                    params[cid].noise_sigma,
                )
                for event in NOISE_EVENTS
            }
            for cid in channel_ids
        }
        self._episode = _Episode(
            seeds=seed_bundle,
            scenario=scenario,
            horizon_hours=horizon_hours,
            total_budget=total_budget,
            channel_ids=channel_ids,
            params={cid: params[cid] for cid in channel_ids},
            tapes=tapes,
            states={cid: ChannelState() for cid in channel_ids},
        )
        empty = Observation(
            hour=0,
            by_channel={
                cid: ChannelObservation(
                    requests=0, impressions=0, unique_reach=0, clicks=0, conversions=0, spend=0.0, ecpm=0.0
                )
                for cid in channel_ids
            },
        )
        return empty, self._info()

    # ------------------------------------------------------------------- step
    def step(self, action: Action | dict) -> tuple[Observation, StepMetrics, bool, StepInfo]:
        ep = self._require_episode()
        if ep.terminated:
            raise RuntimeError("эпизод завершён: вызовите reset перед новым step")
        act = action if isinstance(action, Action) else Action(**action)
        self._validate(ep, act)

        hour = ep.hour
        by_channel: dict[str, ChannelObservation] = {}
        for cid in ep.channel_ids:
            shock = self._active_shocks(ep, cid, hour)
            noise = {event: float(ep.tapes[cid][event][hour]) for event in NOISE_EVENTS}
            obs = step_channel(
                ep.params[cid], ep.states[cid], hour, act.spend_caps[cid], shock, noise, ep.seeds.noise_seed
            )
            _assert_invariants(cid, act.spend_caps[cid], obs)
            by_channel[cid] = obs

        observation = Observation(hour=hour + 1, by_channel=by_channel)
        ep.cumulative_spend += observation.total_spend
        ep.cumulative_clicks += observation.total_clicks
        ep.cumulative_conversions += observation.total_conversions
        ep.hour += 1
        ep.terminated = ep.hour >= ep.horizon_hours
        metrics = StepMetrics(
            cumulative_spend=ep.cumulative_spend,
            cumulative_clicks=ep.cumulative_clicks,
            cumulative_conversions=ep.cumulative_conversions,
            remaining_budget=max(ep.total_budget - ep.cumulative_spend, 0.0),
        )
        info = self._info(terminated_reason="horizon" if ep.terminated else None)
        return observation, metrics, ep.terminated, info

    # ------------------------------------------------------------ extensions
    def inject_shock(self, event: ShockEvent) -> None:
        """Шок из интерфейса во время прогона (расширение контракта, требование презентации)."""
        ep = self._require_episode()
        unknown = set(event.target_channels) - set(ep.channel_ids)
        if unknown:
            raise ValueError(f"шок адресован неизвестным каналам: {sorted(unknown)}")
        ep.injected.append(event)

    @property
    def hour(self) -> int:
        return self._require_episode().hour

    @property
    def remaining_budget(self) -> float:
        ep = self._require_episode()
        return max(ep.total_budget - ep.cumulative_spend, 0.0)

    def debug_snapshot(self) -> dict:
        """Привилегированный снимок для тестов. Стратегии передавать нельзя."""
        ep = self._require_episode()
        return {
            cid: {
                "ecpm_base": p.ecpm_base,
                "base_ctr": p.base_ctr,
                "base_cvr": p.base_cvr,
                "daily_requests": p.daily_requests,
                "unique_pool": p.unique_pool,
                "reached": ep.states[cid].reached,
            }
            for cid, p in ep.params.items()
        }

    # ---------------------------------------------------------------- helpers
    def _require_episode(self) -> _Episode:
        if self._episode is None:
            raise RuntimeError("сначала вызовите reset")
        return self._episode

    def _validate(self, ep: _Episode, action: Action) -> None:
        expected = set(ep.channel_ids)
        got = set(action.spend_caps)
        if got != expected:
            raise ValueError(
                f"action должен содержать ровно активные каналы; лишние {sorted(got - expected)}, "
                f"нет {sorted(expected - got)}"
            )
        remaining = ep.total_budget - ep.cumulative_spend
        if action.total > remaining + 1e-6:
            raise ValueError(
                f"сумма лимитов {action.total:.2f} превышает остаток бюджета {remaining:.2f}"
            )

    @staticmethod
    def _active_shocks(ep: _Episode, channel_id: str, hour: int) -> dict[ShockParameter, float]:
        active: dict[ShockParameter, float] = {}
        for event in [*ep.scenario.events, *ep.injected]:
            if channel_id not in event.target_channels:
                continue
            factor = event.factor_at(hour)
            if factor is None:
                continue
            active[event.parameter] = active.get(event.parameter, 1.0) * factor
        return active

    def _info(self, terminated_reason: str | None = None) -> StepInfo:
        ep = self._require_episode()
        return StepInfo(
            api_version=API_VERSION,
            episode_id=ep.episode_id,
            scenario_id=ep.scenario.scenario_id,
            applied_constraints=["spend<=cap", "reach<=impressions", "conversions<=clicks", "sum(caps)<=remaining"],
            terminated_reason=terminated_reason,
        )


def _assert_invariants(channel_id: str, cap: float, obs: ChannelObservation) -> None:
    if obs.spend > cap + 1e-6:
        raise AssertionError(f"{channel_id}: spend {obs.spend} > cap {cap}")
    if obs.impressions > obs.requests:
        raise AssertionError(f"{channel_id}: impressions > requests")
    if obs.unique_reach > obs.impressions:
        raise AssertionError(f"{channel_id}: reach > impressions")
    if obs.clicks > obs.impressions or obs.conversions > obs.clicks:
        raise AssertionError(f"{channel_id}: воронка нарушена")
