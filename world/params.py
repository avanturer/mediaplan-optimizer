"""Истинные (скрытые) параметры каналов эпизода.

Выбираются из ``world_seed`` внутри диапазонов публичного каталога или рядом
с ними (контракт мира, §11). Именно это расхождение между прайс-листом и
реальностью делает нужным сервис исполнения: план по каталогу гарантированно
разойдётся с фактом.
"""

from dataclasses import dataclass, field

import numpy as np

from contracts.catalog import CatalogChannel, ChannelFamily, PublicCatalog
from world.config import load_assumptions, range_of, value_of

HOURS_IN_WEEK = 168


@dataclass(frozen=True)
class HiddenChannelParams:
    channel_id: str
    family: ChannelFamily
    daily_requests: float  # средний дневной инвентарь (контактов в сутки)
    hourly_profile: np.ndarray  # 168 долей, сумма внутри каждых суток = 1
    ecpm_base: float  # медиана цены тысячи показов
    price_sigma: float  # разброс лог-нормального ландшафта цены
    base_ctr: float
    base_cvr: float
    unique_pool: float  # сколько уникальных людей можно охватить за кампанию
    fatigue_delta: float  # ctr(f) = ctr0 / (1 + delta·(f − 1))
    noise_rho: float
    noise_sigma: float
    # только для SMS
    sms_price: float = 0.0
    sms_deliverability: float = 1.0
    sms_base_size: int = 0
    sms_cooldown_days: int = 1
    sms_send_hours: tuple[int, int] = (9, 21)
    extras: dict = field(default_factory=dict)


def _draw_near_range(rng: np.random.Generator, lo: float, hi: float, outside: float) -> float:
    """Равномерно внутри диапазона, расширенного на долю ``outside`` с каждой стороны."""
    width = hi - lo
    return float(rng.uniform(lo - outside * width, hi + outside * width))


def _hourly_profile(rng: np.random.Generator, assumptions: dict) -> np.ndarray:
    """Двугорбый суточный профиль с утренним и вечерним пиками, выходные мягче.

    Форма из практики медиапланирования (config/assumptions.yaml, hourly_demand);
    амплитуда выбирается из диапазона и скрыта от планировщика.
    """
    peak_lo, peak_hi = range_of(assumptions["hourly_demand"]["weekday_peak_ratio"])
    trough_lo, trough_hi = range_of(assumptions["hourly_demand"]["night_trough_ratio"])
    peak = rng.uniform(peak_lo, peak_hi)
    trough = rng.uniform(trough_lo, trough_hi)
    hours = np.arange(24)
    morning = np.exp(-((hours - 10) ** 2) / (2 * 2.5**2))
    evening = np.exp(-((hours - 20) ** 2) / (2 * 2.5**2))
    shape = trough + (peak - trough) * (0.6 * morning + evening) / 1.0
    profile = np.empty(HOURS_IN_WEEK)
    for day in range(7):
        weekend = day >= 5
        daily = shape * (0.85 if weekend else 1.0)
        if weekend:
            # в выходные утренний пик сдвигается позже и сглаживается
            daily = np.roll(daily, 1)
        profile[day * 24 : (day + 1) * 24] = daily / daily.sum()
    return profile


def draw_hidden_params(catalog: PublicCatalog, world_seed: int) -> dict[str, HiddenChannelParams]:
    assumptions = load_assumptions()
    rng = np.random.default_rng(world_seed)
    outside = value_of(assumptions["hidden_parameter_draw"]["outside_range_share"])
    multiplier = value_of(assumptions["audience_scale"]["campaign_audience_multiplier"])
    delta_lo, delta_hi = range_of(assumptions["frequency_fatigue"]["delta"])
    rho = value_of(assumptions["noise"]["ar1_rho"])
    sigma = value_of(assumptions["noise"]["ar1_sigma"])
    params: dict[str, HiddenChannelParams] = {}

    for channel in catalog.channels:
        params[channel.channel_id] = _draw_channel(
            rng, channel, assumptions, outside, multiplier, (delta_lo, delta_hi), rho, sigma
        )
    return params


def _draw_channel(
    rng: np.random.Generator,
    channel: CatalogChannel,
    assumptions: dict,
    outside: float,
    multiplier: float,
    delta_range: tuple[float, float],
    rho: float,
    sigma: float,
) -> HiddenChannelParams:
    ecpm = _draw_near_range(rng, *channel.expected_ecpm_range, outside)
    ctr = _draw_near_range(rng, *channel.expected_ctr_range, outside)
    cvr = _draw_near_range(rng, *channel.expected_cvr_range, outside)
    daily_unique = _draw_near_range(rng, *channel.daily_unique_capacity_band, outside)
    profile = _hourly_profile(rng, assumptions)
    fatigue = float(rng.uniform(*delta_range))

    if channel.family is ChannelFamily.DIRECT:
        assert channel.sms is not None
        return HiddenChannelParams(
            channel_id=channel.channel_id,
            family=channel.family,
            daily_requests=channel.sms.base_size / channel.sms.cooldown_days,
            hourly_profile=profile,
            ecpm_base=channel.sms.price_per_message_rub * 1000,
            price_sigma=0.0,
            base_ctr=max(ctr, 1e-4),
            base_cvr=max(cvr, 1e-4),
            unique_pool=float(channel.sms.base_size),
            fatigue_delta=fatigue,
            noise_rho=rho,
            noise_sigma=sigma * 0.5,
            sms_price=channel.sms.price_per_message_rub,
            sms_deliverability=channel.sms.deliverability,
            sms_base_size=channel.sms.base_size,
            sms_cooldown_days=channel.sms.cooldown_days,
            sms_send_hours=channel.sms.send_hours,
        )

    family_key = channel.family.value
    kappa_lo, kappa_hi = range_of(assumptions["contacts_per_user_per_day"]["by_family"][family_key])
    kappa = float(rng.uniform(kappa_lo, kappa_hi))
    sigma_lo, sigma_hi = range_of(assumptions["price_sigma_by_family"][family_key])
    price_sigma = float(rng.uniform(sigma_lo, sigma_hi))
    return HiddenChannelParams(
        channel_id=channel.channel_id,
        family=channel.family,
        daily_requests=daily_unique * kappa,
        hourly_profile=profile,
        ecpm_base=max(ecpm, 1.0),
        price_sigma=price_sigma,
        base_ctr=max(ctr, 1e-4),
        base_cvr=max(cvr, 1e-4),
        unique_pool=daily_unique * multiplier,
        fatigue_delta=fatigue,
        noise_rho=rho,
        noise_sigma=sigma,
    )
