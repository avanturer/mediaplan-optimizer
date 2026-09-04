"""Движок одного часа: как лимит расхода превращается в показы, охват, клики и конверсии.

Порядок разыгрывания (контракт мира, §6): трафик → цена и доступность → охват и
отклик → обновление скрытого состояния. Три семейства поведения:

- ``auction`` и ``marketplace``: цена инвентаря распределена лог-нормально
  (bid landscape, arXiv:2001.06587); лимит выкупает самую дешёвую долю
  инвентаря, поэтому средняя цена растёт с выкупом, а показы упираются в
  потолок. У маркетплейсов инвентарь уже, а цена растёт резче (price_sigma).
- ``direct`` (SMS): цена фиксированная, аукциона нет, но есть окно отправки,
  дневная квота базы и доставляемость.

Охват: модель случайного размещения показов по пулу,
``ΔR = (N − R)·(1 − exp(−n/N))``. Усталость: ``ctr = ctr0 / (1 + δ·(f − 1))``.
"""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from contracts.catalog import ChannelFamily
from contracts.simulation import ChannelObservation, ShockParameter
from world.params import HiddenChannelParams
from world.rng import keyed_rng

_NORMAL = NormalDist()


@dataclass
class ChannelState:
    """Скрытое состояние канала, переходит из часа в час."""

    reached: float = 0.0  # накопленный уникальный охват
    cum_impressions: float = 0.0
    sms_sent_in_cycle: int = 0  # сколько сообщений отправлено в текущем цикле cooldown

    @property
    def frequency(self) -> float:
        return self.cum_impressions / self.reached if self.reached > 0 else 1.0


def lognormal_partial_mean(median: float, sigma: float, share: float) -> float:
    """Средняя цена самой дешёвой доли ``share`` инвентаря при LogNormal(ln median, sigma).

    E[P | P ≤ q_share] = median·exp(σ²/2)·Φ(Φ⁻¹(share) − σ) / share.
    """
    if share <= 0:
        return median * float(np.exp(-sigma * sigma / 2))
    if share >= 1:
        return median * float(np.exp(sigma * sigma / 2))
    z = _NORMAL.inv_cdf(share)
    return median * float(np.exp(sigma * sigma / 2)) * _NORMAL.cdf(z - sigma) / share


def buy_impressions(cap_rub: float, available: float, ecpm_median: float, sigma: float) -> tuple[float, float]:
    """Сколько показов покупает лимит ``cap_rub`` из ``available`` и по какой средней цене.

    Ищем долю выкупа f, при которой расход равен лимиту; расход монотонно растёт
    по f, поэтому бисекция. Если денег хватает на всё, f = 1 и остаток лимита не
    тратится: это и есть «исчерпание ёмкости» без единой лишней константы.
    """
    if cap_rub <= 0 or available <= 0:
        return 0.0, 0.0

    def spend_at(share: float) -> float:
        return available * share / 1000 * lognormal_partial_mean(ecpm_median, sigma, share)

    if spend_at(1.0) <= cap_rub:
        return available, lognormal_partial_mean(ecpm_median, sigma, 1.0)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if spend_at(mid) > cap_rub:
            hi = mid
        else:
            lo = mid
    share = lo
    return available * share, lognormal_partial_mean(ecpm_median, sigma, max(share, 1e-9))


def reach_increment(pool: float, reached: float, impressions: float) -> float:
    if pool <= 0 or impressions <= 0:
        return 0.0
    remaining = max(pool - reached, 0.0)
    return remaining * (1 - float(np.exp(-impressions / pool)))


def step_channel(
    params: HiddenChannelParams,
    state: ChannelState,
    hour: int,
    cap_rub: float,
    shock: dict[ShockParameter, float],
    noise: dict[str, float],
    noise_seed: int,
) -> ChannelObservation:
    """Разыгрывает час для одного канала и обновляет его состояние на месте."""
    if shock.get(ShockParameter.PAUSE) is not None:
        return ChannelObservation(
            requests=0, impressions=0, unique_reach=0, clicks=0, conversions=0, spend=0.0, ecpm=0.0
        )

    hour_of_week = hour % 168
    demand_factor = shock.get(ShockParameter.DEMAND, 1.0) * noise["traffic"]
    inventory_factor = shock.get(ShockParameter.INVENTORY, 1.0)
    price_factor = shock.get(ShockParameter.ECPM, 1.0) * noise["price"]
    ctr_factor = shock.get(ShockParameter.CTR, 1.0) * noise["ctr"]
    cvr_factor = shock.get(ShockParameter.CVR, 1.0) * noise["cvr"]

    if params.family is ChannelFamily.DIRECT:
        return _step_sms(params, state, hour, cap_rub, inventory_factor, ctr_factor, cvr_factor, noise_seed)

    lam = params.daily_requests * params.hourly_profile[hour_of_week] * demand_factor * inventory_factor
    requests = int(keyed_rng(noise_seed, hour, params.channel_id, "traffic").poisson(max(lam, 0.0)))
    impressions_f, avg_price = buy_impressions(
        cap_rub, float(requests), params.ecpm_base * price_factor, params.price_sigma
    )
    impressions = int(round(impressions_f))
    spend = impressions / 1000 * avg_price
    if spend > cap_rub:  # округление показов не должно нарушать инвариант spend ≤ cap
        spend = cap_rub
    ecpm = spend / impressions * 1000 if impressions else 0.0

    new_reach = reach_increment(params.unique_pool, state.reached, impressions)
    unique_reach = min(int(round(new_reach)), impressions)
    state.reached += unique_reach
    state.cum_impressions += impressions

    ctr = params.base_ctr * ctr_factor / (1 + params.fatigue_delta * max(state.frequency - 1, 0.0))
    ctr = min(max(ctr, 0.0), 1.0)
    clicks = int(keyed_rng(noise_seed, hour, params.channel_id, "click").binomial(impressions, ctr))
    cvr = min(max(params.base_cvr * cvr_factor, 0.0), 1.0)
    conversions = int(keyed_rng(noise_seed, hour, params.channel_id, "conversion").binomial(clicks, cvr))

    return ChannelObservation(
        requests=requests,
        impressions=impressions,
        unique_reach=unique_reach,
        clicks=clicks,
        conversions=conversions,
        spend=spend,
        ecpm=ecpm,
    )


def _step_sms(
    params: HiddenChannelParams,
    state: ChannelState,
    hour: int,
    cap_rub: float,
    inventory_factor: float,
    ctr_factor: float,
    cvr_factor: float,
    noise_seed: int,
) -> ChannelObservation:
    """SMS: пакет отправок в окне 9–21, не больше дневной квоты базы и остатка цикла."""
    hour_of_day = hour % 24
    start, end = params.sms_send_hours
    window = max(end - start, 1)
    cycle_hours = params.sms_cooldown_days * 24
    if hour % cycle_hours == 0:
        state.sms_sent_in_cycle = 0
    if not (start <= hour_of_day < end):
        return ChannelObservation(
            requests=0, impressions=0, unique_reach=0, clicks=0, conversions=0, spend=0.0, ecpm=0.0
        )
    daily_quota = params.sms_base_size / params.sms_cooldown_days * inventory_factor
    hourly_quota = int(daily_quota / window)
    left_in_cycle = max(int(params.sms_base_size * inventory_factor) - state.sms_sent_in_cycle, 0)
    requests = min(hourly_quota, left_in_cycle)
    affordable = int(cap_rub // params.sms_price) if params.sms_price > 0 else 0
    sent = max(min(requests, affordable), 0)
    delivered = int(keyed_rng(noise_seed, hour, params.channel_id, "delivery").binomial(sent, params.sms_deliverability))
    spend = sent * params.sms_price
    state.sms_sent_in_cycle += sent

    new_reach = reach_increment(params.unique_pool, state.reached, delivered)
    unique_reach = min(int(round(new_reach)), delivered)
    state.reached += unique_reach
    state.cum_impressions += delivered

    ctr = params.base_ctr * ctr_factor / (1 + params.fatigue_delta * max(state.frequency - 1, 0.0))
    ctr = min(max(ctr, 0.0), 1.0)
    clicks = int(keyed_rng(noise_seed, hour, params.channel_id, "click").binomial(delivered, ctr))
    cvr = min(max(params.base_cvr * cvr_factor, 0.0), 1.0)
    conversions = int(keyed_rng(noise_seed, hour, params.channel_id, "conversion").binomial(clicks, cvr))
    ecpm = spend / delivered * 1000 if delivered else 0.0
    return ChannelObservation(
        requests=requests,
        impressions=delivered,
        unique_reach=unique_reach,
        clicks=clicks,
        conversions=conversions,
        spend=spend,
        ecpm=ecpm,
    )
