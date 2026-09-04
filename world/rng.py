"""Экзогенная случайность по стабильным ключам (контракт мира, §9).

Одного последовательного генератора недостаточно: разные стратегии принимают
разные действия, ветвящаяся логика потребляет разное число случайных чисел, и
парное сравнение разрушается. Поэтому каждое случайное значение адресуется
ключом ``(noise_seed, hour, channel_id, event_type)``: две стратегии на одном
``noise_seed`` видят одну и ту же «ленту» событий, меняется только policy.
Это common random numbers из практики имитационного моделирования.
"""

import hashlib

import numpy as np


def keyed_rng(noise_seed: int, hour: int, channel_id: str, event_type: str) -> np.random.Generator:
    """Детерминированный генератор для одного события одного часа одного канала."""
    key = f"{noise_seed}|{hour}|{channel_id}|{event_type}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def ar1_tape(
    noise_seed: int,
    channel_id: str,
    event_type: str,
    horizon_hours: int,
    rho: float,
    sigma: float,
) -> np.ndarray:
    """Лента мультипликативного шума exp(x_t), x_t = rho·x_{t-1} + eps_t.

    Автокоррелированный шум, а не белый: белый усредняется за сутки, и любой
    контроллер выигрывает случайно; AR(1) создаёт «полосы невезения», на
    которых видна разница между стратегиями. Лента строится целиком на
    ``reset`` и не зависит от действий.
    """
    stationary_sd = sigma / np.sqrt(max(1 - rho * rho, 1e-9))
    x = np.empty(horizon_hours)
    prev = keyed_rng(noise_seed, -1, channel_id, event_type).normal(0.0, stationary_sd)
    for hour in range(horizon_hours):
        eps = keyed_rng(noise_seed, hour, channel_id, event_type).normal(0.0, sigma)
        prev = rho * prev + eps
        x[hour] = prev
    # центрируем, чтобы E[exp(x)] ≈ 1 и шум не сдвигал средний уровень
    return np.exp(x - stationary_sd**2 / 2)
