"""Симулятор рынка: reset / step / inject_shock.

Реализует публичный контракт модели мира (docs/world/WORLD_MODEL.md, §8):
один вызов ``step`` разыгрывает один час для всех активных каналов и
возвращает только агрегаты. Скрытое состояние наружу не выходит, ``info``
содержит лишь безопасные идентификаторы и сообщения валидации.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

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
from contracts.targeting import AudienceTargeting
from world.audience import Audience
from world.catalog_builder import build_catalog
from world.competition import competition_tapes
from world.engine import ChannelState, step_channel
from world.params import HiddenChannelParams, draw_hidden_params
from world.rng import ar1_tape
from world.scenarios import get_scenario
from world.settings import WorldSettings
from world.targeting import catalog_for_targeting

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
    audience: Audience
    pressure: dict[str, np.ndarray]
    settings: WorldSettings
    manifest: dict
    config_hash: str
    states: dict[str, ChannelState] = field(default_factory=dict)
    hour: int = 0
    cumulative_spend: float = 0.0
    cumulative_clicks: int = 0
    cumulative_conversions: int = 0
    cumulative_reach: int = 0
    injected: list[ShockEvent] = field(default_factory=list)
    terminated: bool = False

    @property
    def episode_id(self) -> str:
        return self.config_hash[:12]


class Simulator:
    """Мир восьми каналов с шагом в один час."""

    def __init__(self, catalog: PublicCatalog | None = None, catalog_seed: int = 0,
                 settings: WorldSettings | None = None) -> None:
        self.catalog = catalog or build_catalog(catalog_seed)
        self.settings = (settings or WorldSettings()).model_copy(deep=True)
        self._episode: _Episode | None = None

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        seed_bundle: SeedBundle,
        scenario_id: str | Scenario = "stable",
        horizon_hours: int = 21 * 24,
        total_budget: float = 1_200_000.0,
        channel_ids: list[str] | None = None,
        targeting: AudienceTargeting | None = None,
    ) -> tuple[Observation, StepInfo]:
        scenario = (scenario_id if isinstance(scenario_id, Scenario) else get_scenario(scenario_id)).model_copy(deep=True)
        if horizon_hours <= 0 or not isinstance(horizon_hours, int):
            raise ValueError("horizon_hours должен быть положительным целым")
        if not math.isfinite(total_budget) or total_budget < 0:
            raise ValueError("total_budget должен быть конечным и неотрицательным")
        channel_ids = list(self.catalog.channel_ids if channel_ids is None else channel_ids)
        if not channel_ids or len(channel_ids) != len(set(channel_ids)):
            raise ValueError("активные каналы должны быть непустыми и уникальными")
        unknown = set(channel_ids) - set(self.catalog.channel_ids)
        if unknown:
            raise ValueError(f"неизвестные каналы: {sorted(unknown)}")
        for event in scenario.events:
            self._validate_shock(event)
        settings = self.settings.model_copy(deep=True)
        for rival in settings.competitors:
            if set(rival.channel_advantages) - set(self.catalog.channel_ids):
                raise ValueError("неизвестные каналы в преимуществах конкурента")
        targeted_catalog = catalog_for_targeting(self.catalog, targeting or self.catalog.targeting)
        params = draw_hidden_params(targeted_catalog, seed_bundle.world_seed)
        audience = Audience({cid: p.unique_pool for cid, p in params.items()}, settings)
        manifest = {
            "api_version": API_VERSION,
            "world_fingerprint": _world_fingerprint(),
            "catalog": targeted_catalog.model_dump(mode="json"),
            "settings": settings.model_dump(mode="json"),
            "seed_bundle": seed_bundle.model_dump(),
            "scenario": scenario.model_dump(mode="json"),
            "horizon_hours": horizon_hours,
            "total_budget": total_budget,
            "channel_ids": channel_ids,
            "injected": [],
        }
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
            seeds=seed_bundle.model_copy(deep=True),
            scenario=scenario,
            horizon_hours=horizon_hours,
            total_budget=total_budget,
            channel_ids=channel_ids,
            params={cid: params[cid] for cid in channel_ids},
            tapes=tapes,
            audience=audience,
            pressure=competition_tapes(settings, seed_bundle.noise_seed, channel_ids, horizon_hours),
            settings=settings,
            manifest=manifest,
            config_hash=_hash_manifest(manifest),
            states={cid: ChannelState() for cid in channel_ids},
        )
        empty = Observation(
            hour=0,
            deduplicated_reach=0,
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
            pressure = float(ep.pressure[cid][hour])
            shock[ShockParameter.ECPM] = shock.get(ShockParameter.ECPM, 1.0) * (1 + pressure)
            shock[ShockParameter.INVENTORY] = shock.get(ShockParameter.INVENTORY, 1.0) / (1 + pressure)
            shock[ShockParameter.CVR] = shock.get(ShockParameter.CVR, 1.0) / (1 + pressure)
            noise = {event: float(ep.tapes[cid][event][hour]) for event in NOISE_EVENTS}
            obs = step_channel(
                ep.params[cid], ep.states[cid], hour, act.spend_caps[cid], shock, noise, ep.seeds.noise_seed, ep.settings
            )
            _assert_invariants(cid, act.spend_caps[cid], obs)
            by_channel[cid] = obs

        reach = ep.audience.step({cid: ep.states[cid].human_impressions_last_hour for cid in ep.channel_ids})
        observation = Observation(hour=hour + 1, by_channel=by_channel, deduplicated_reach=reach)
        ep.cumulative_reach += reach
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
            cumulative_reach=ep.cumulative_reach,
        )
        info = self._info(terminated_reason="horizon" if ep.terminated else None)
        return observation, metrics, ep.terminated, info

    # ------------------------------------------------------------ extensions
    def inject_shock(self, event: ShockEvent) -> None:
        """Шок из интерфейса во время прогона (расширение контракта, требование презентации)."""
        ep = self._require_episode()
        self._validate_shock(event)
        unknown = set(event.target_channels) - set(ep.channel_ids)
        if unknown:
            raise ValueError(f"шок адресован неизвестным каналам: {sorted(unknown)}")
        if event.start_hour < ep.hour:
            raise ValueError("нельзя добавить шок в уже завершённый час")
        ep.injected.append(event.model_copy(deep=True))
        ep.manifest["injected"] = [e.model_dump(mode="json") for e in ep.injected]
        ep.config_hash = _hash_manifest(ep.manifest)

    def export_manifest(self) -> dict:
        """Привилегированный рецепт повтора с начала, НЕ public info и НЕ вход стратегии."""
        return json.loads(json.dumps(self._require_episode().manifest))

    @classmethod
    def from_manifest(cls, manifest: dict) -> "Simulator":
        """Создаёт и сбрасывает эпизод; для повтора нужны те же действия."""
        if manifest["api_version"] != API_VERSION or manifest["world_fingerprint"] != _world_fingerprint():
            raise ValueError("версия кода или конфигурация мира отличается от сохранённой")
        sim = cls(PublicCatalog.model_validate(manifest["catalog"]), settings=WorldSettings.model_validate(manifest["settings"]))
        sim.reset(SeedBundle.model_validate(manifest["seed_bundle"]), Scenario.model_validate(manifest["scenario"]),
                  horizon_hours=manifest["horizon_hours"], total_budget=manifest["total_budget"],
                  channel_ids=manifest["channel_ids"])
        for event in manifest["injected"]:
            sim.inject_shock(ShockEvent.model_validate(event))
        return sim

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
    def _validate_shock(self, event: ShockEvent) -> None:
        for cid in event.target_channels:
            if cid not in self.catalog.channel_ids:
                raise ValueError(f"неизвестный канал шока: {cid}")
            if event.parameter == ShockParameter.FRAUD and cid != "programmatic":
                raise ValueError("fraud поддерживается только для programmatic")
            if event.parameter == ShockParameter.SMS_WEEKLY_LIMIT and self.catalog.by_id(cid).sms is None:
                raise ValueError("sms_weekly_limit поддерживается только для SMS")

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
            config_hash=ep.config_hash,
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
    if obs.verified_impressions > obs.impressions:
        raise AssertionError(f"{channel_id}: verified_impressions > impressions")


def _hash_manifest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _world_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "world").glob("*.py")) + sorted((root / "config").glob("*.yaml"))
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in paths)).hexdigest()
