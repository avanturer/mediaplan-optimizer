"""Внешние компании: ограниченное экзогенное давление с преимуществами по каналам."""

import numpy as np

from world.rng import ar1_tape
from world.settings import WorldSettings


def competition_tapes(settings: WorldSettings, noise_seed: int, channels: list[str], hours: int) -> dict[str, np.ndarray]:
    pressure = {cid: np.zeros(hours) for cid in channels}
    for rival in sorted(settings.competitors, key=lambda c: c.competitor_id):
        for cid in channels:
            advantage = rival.channel_advantages.get(cid, 0.0)
            if not advantage or not rival.strength:
                continue
            # Отдельный ключ и плавный AR(1); множитель всегда в [1-v, 1+v].
            tape = ar1_tape(noise_seed, cid, f"competitor:{rival.competitor_id}", hours, 0.9, 0.25)
            activity = 1 + rival.volatility * np.tanh(np.log(tape))
            start = rival.start_hour
            end = rival.end_hour if rival.end_hour is not None else hours
            pressure[cid][start:end] += rival.strength * advantage * activity[start:end]
    # Ограниченный рынок: рост CPM не более ×3, доступность не ниже 1/3.
    return {cid: np.minimum(values, 2.0) for cid, values in pressure.items()}
