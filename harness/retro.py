"""Сбор ретро-истории: пробные кампании через публичный reset/step.

Каждый пробный эпизод это один день с постоянным часовым лимитом на уровне
``level × ref_daily_spend`` для всех каналов сразу, начатый с чистого
состояния (частота около 1). Лестница уровней даёт точки кривой
«дневной бюджет → отклик», уровни выше потолка показывают, где лимит
перестаёт связывать. Один длинный эпизод даёт профиль по часам недели.

Пробы идут на отдельных ``world_seed`` (ретро-кампании прошлого), поэтому
кривые расходятся с миром оцениваемой кампании ровно так, как прогноз
кабинета расходится с фактом. Мозг получает только действия и наблюдения.
"""

from brain.config import PUBLIC_CONTACTS_PER_USER, RETRO_WORLD_SEEDS
from brain.config import RETRO_LEVELS as LEVELS
from contracts import Action, PublicCatalog, RetroEpisode, RetroHistory, SeedBundle
from world.simulator import Simulator


def reference_daily_spend(catalog: PublicCatalog, channel_id: str) -> float:
    """Верхняя оценка дневного расхода канала по публичным числам каталога."""
    ch = catalog.by_id(channel_id)
    if ch.sms is not None:
        return ch.sms.base_size / ch.sms.cooldown_days * ch.sms.price_per_message_rub * 1.5
    return ch.daily_unique_capacity_band[1] * PUBLIC_CONTACTS_PER_USER * ch.expected_ecpm_range[1] / 1000


def collect_retro_history(
    catalog: PublicCatalog,
    channel_ids: list[str] | None = None,
    catalog_seed: int = 0,
    world_seeds: tuple[int, ...] = RETRO_WORLD_SEEDS,
    levels: tuple[float, ...] = LEVELS,
    profile_days: int = 7,
) -> RetroHistory:
    channel_ids = list(channel_ids or catalog.channel_ids)
    sim = Simulator(catalog)
    episodes: list[RetroEpisode] = []
    refs = {cid: reference_daily_spend(catalog, cid) for cid in channel_ids}

    for world_seed in world_seeds:
        for level in levels:
            horizon = 24
            caps = {cid: refs[cid] * level / 24 for cid in channel_ids}
            episodes.append(
                _run_probe(sim, catalog_seed, world_seed, horizon, caps, channel_ids, f"probe_{world_seed}_{level}")
            )
        # длинный эпизод на среднем уровне: профиль по часам недели
        caps = {cid: refs[cid] * 0.15 / 24 for cid in channel_ids}
        episodes.append(
            _run_probe(sim, catalog_seed, world_seed, profile_days * 24, caps, channel_ids, f"profile_{world_seed}")
        )
    return RetroHistory(catalog_id=catalog.catalog_id, episodes=episodes)


def _run_probe(
    sim: Simulator,
    catalog_seed: int,
    world_seed: int,
    horizon: int,
    caps: dict[str, float],
    channel_ids: list[str],
    episode_id: str,
) -> RetroEpisode:
    total_budget = sum(caps.values()) * horizon * 1.01 + 1.0
    seeds = SeedBundle(catalog_seed=catalog_seed, world_seed=world_seed, noise_seed=world_seed * 7 + horizon)
    sim.reset(seeds, "stable", horizon_hours=horizon, total_budget=total_budget, channel_ids=channel_ids)
    actions, observations = [], []
    for _ in range(horizon):
        action = Action(spend_caps=dict(caps))
        obs, _, _, _ = sim.step(action)
        actions.append(action)
        observations.append(obs)
    return RetroEpisode(episode_id=episode_id, horizon_hours=horizon, actions=actions, observations=observations)
