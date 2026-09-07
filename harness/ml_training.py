"""Офлайн-подготовка данных ML на отдельных мирах через reset/step.

Расписание исторических экспериментов используется ТОЛЬКО как label
классификатора. brain получает массив признаков из наблюдений, не сценарии.
"""

import numpy as np

from brain.curves import ResponseCurve
from brain.ml import MLBundle, QualityModel, QualityWindow, ReachModel, fit_response_curves
from contracts import (
    Action,
    PublicCatalog,
    RetroEpisode,
    RetroHistory,
    Scenario,
    SeedBundle,
    ShockEvent,
    ShockParameter,
)
from harness.retro import collect_retro_history, reference_daily_spend
from world import Simulator

TRAIN_SEEDS = tuple(range(2000, 2024))
VALIDATION_SEEDS = tuple(range(2100, 2108))


def collect_ml_history(catalog: PublicCatalog, seeds: tuple[int, ...]) -> tuple[RetroHistory, np.ndarray, np.ndarray]:
    episodes, features, labels = [], [], []
    families = (None, ShockParameter.FRAUD, ShockParameter.CTR, ShockParameter.CVR,
                ShockParameter.ECPM, ShockParameter.INVENTORY)
    refs = {cid: reference_daily_spend(catalog, cid) for cid in catalog.channel_ids}
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        parameter = families[index % len(families)]
        non_sms = [cid for cid in catalog.channel_ids if catalog.by_id(cid).sms is None]
        target = "programmatic" if parameter == ShockParameter.FRAUD else str(rng.choice(non_sms))
        start = int(rng.integers(32, 65))
        multiplier = (float(rng.uniform(5, 20)) if parameter == ShockParameter.FRAUD else
                      float(rng.uniform(1.7, 3)) if parameter == ShockParameter.ECPM else float(rng.uniform(0.15, 0.55)))
        events = [] if parameter is None else [ShockEvent(start_hour=start, duration_hours=36,
            target_channels=[target], parameter=parameter, multiplier=multiplier)]
        sim = Simulator(catalog)
        sim.reset(SeedBundle(world_seed=seed, noise_seed=seed + 30_000),
                  Scenario(scenario_id="historical_training", events=events), horizon_hours=120, total_budget=1e10)
        windows = {cid: QualityWindow() for cid in catalog.channel_ids}
        actions, observations = [], []
        for hour in range(120):
            if hour % 24 == 0:
                levels = rng.choice([0.0, 0.05, 0.15, 0.4, 0.8, 1.5], size=len(refs))
                caps = {cid: refs[cid] * float(level) / 24 for cid, level in zip(refs, levels, strict=True)}
            action = Action(spend_caps=caps)
            obs, _, _, _ = sim.step(action)
            actions.append(action)
            observations.append(obs)
            for cid, row in obs.by_channel.items():
                x = windows[cid].update(row, caps[cid])
                if x is not None:
                    # После перехода требуется полное недавнее окно наблюдений.
                    if parameter is not None and cid == target and (start <= hour < start + 6 or start + 36 <= hour < start + 60):
                        continue
                    features.append(x)
                    labels.append(float(parameter is not None and cid == target and start + 6 <= hour < start + 36))
        episodes.append(RetroEpisode(episode_id=f"ml_history_{index}", horizon_hours=120,
                                     actions=actions, observations=observations))
    return RetroHistory(catalog_id=catalog.catalog_id, episodes=episodes), np.array(features), np.array(labels)


def train_ml_bundle(catalog: PublicCatalog, curves: dict[str, ResponseCurve],
                    response_history: RetroHistory | None = None) -> MLBundle:
    response_history = response_history or collect_retro_history(catalog)
    history, x, y = collect_ml_history(catalog, TRAIN_SEEDS)
    _, vx, vy = collect_ml_history(catalog, VALIDATION_SEEDS)
    quality = QualityModel.fit(x, y, vx, vy)
    return MLBundle(catalog.catalog_id, fit_response_curves(response_history, curves),
                    ReachModel.fit(history, catalog), quality,
                    {"version": "ml-v1", "training_worlds": len(TRAIN_SEEDS),
                     "validation_worlds": len(VALIDATION_SEEDS), "quality_training_rows": len(x),
                     "quality_validation_rows": len(vx), "positive_training_rows": int(y.sum()),
                     "response_episodes": len(response_history.episodes), "synthetic": True})
