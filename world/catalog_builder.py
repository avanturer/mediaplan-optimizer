"""Сборка публичного каталога из реестра бенчмарков и допущений.

Каталог это «прайс-лист»: диапазоны ожиданий по CPM, CTR, CR и ёмкости.
Он строится из ``config/benchmarks.yaml`` (числа со ссылками) и
``config/assumptions.yaml`` (числа с пометкой «наше допущение»). Истинные
параметры эпизода мир выбирает отдельно, внутри этих диапазонов или рядом.

``catalog_seed`` слегка сдвигает середины диапазонов, чтобы каталог тоже был
воспроизводимой случайной величиной, как требует кейс (отдельный seed для
каталога и отдельный для шума кампании).
"""

import hashlib
import json

import numpy as np

from contracts.catalog import CatalogChannel, ChannelFamily, PublicCatalog, SmsPublic
from world.config import load_assumptions, load_benchmarks, range_of, value_of

# channel_id контракта мира → ключ в реестре бенчмарков
CHANNEL_MAP: list[tuple[str, str, ChannelFamily]] = [
    ("social_1", "social_video", ChannelFamily.AUCTION),
    ("social_2", "social_feed", ChannelFamily.AUCTION),
    ("social_3", "social_banner", ChannelFamily.AUCTION),
    ("programmatic", "programmatic", ChannelFamily.AUCTION),
    ("marketplace_1", "marketplace_cpc", ChannelFamily.MARKETPLACE),
    ("marketplace_2", "marketplace_cpm", ChannelFamily.MARKETPLACE),
    ("marketplace_3", "marketplace_premium", ChannelFamily.MARKETPLACE),
    ("sms", "sms", ChannelFamily.DIRECT),
]

# Относительная ширина диапазона, если в источнике дана только медиана.
# Допущение: ±20 % соответствует разбросу квартилей в тех же отчётах click.ru.
DEFAULT_HALF_WIDTH = 0.20


def _band(mid: float, half_width: float = DEFAULT_HALF_WIDTH) -> tuple[float, float]:
    return mid * (1 - half_width), mid * (1 + half_width)


def _ctr_range(node: dict, fallback: dict | None) -> tuple[float, float]:
    if node.get("value") is not None:
        if node.get("p25") is not None and node.get("p75") is not None:
            return float(node["p25"]), float(node["p75"])
        return _band(float(node["value"]))
    if fallback is None:
        raise ValueError("нет ни бенчмарка CTR, ни допущения")
    return range_of(fallback)


def _ecpm_mid(bench: dict, ctr_mid: float) -> float:
    """eCPM канала: прямой бенчмарк CPM, либо CPC × CTR × 1000 для CPC-инвентаря."""
    if bench.get("cpm_rub", {}).get("value") is not None:
        return value_of(bench["cpm_rub"])
    cpc = value_of(bench["cpc_rub"])
    return cpc * ctr_mid * 1000


def build_catalog(catalog_seed: int = 0) -> PublicCatalog:
    benchmarks = load_benchmarks()["channels"]
    assumptions = load_assumptions()
    rng = np.random.default_rng(catalog_seed)
    channels: list[CatalogChannel] = []

    for channel_id, bench_key, family in CHANNEL_MAP:
        bench = benchmarks[bench_key]
        kind = bench["kind"]
        jitter = float(rng.lognormal(0.0, 0.05))  # небольшой сдвиг середин от catalog_seed

        if family is ChannelFamily.DIRECT:
            sms = SmsPublic(
                price_per_message_rub=value_of(bench["price_per_message_rub"]) * jitter,
                deliverability=value_of(bench["deliverability"]),
                base_size=int(
                    value_of(assumptions["audience_scale"]["daily_unique_audience_by_kind"]["sms"])
                    * assumptions["audience_scale"]["campaign_audience_multiplier"]["value"]
                ),
                cooldown_days=7,
            )
            ctr_mid = value_of(bench["ctr"])
            cvr_lo, cvr_hi = range_of(assumptions["conversion_rate"]["by_kind"]["sms"])
            channels.append(
                CatalogChannel(
                    channel_id=channel_id,
                    family=family,
                    display_name=bench["display_name"],
                    expected_ecpm_range=_band(sms.price_per_message_rub * 1000),
                    expected_ctr_range=_band(ctr_mid),
                    expected_cvr_range=(cvr_lo, cvr_hi),
                    daily_unique_capacity_band=(
                        int(sms.base_size / sms.cooldown_days * 0.7),
                        int(sms.base_size / sms.cooldown_days * 1.3),
                    ),
                    sms=sms,
                    benchmark_sources=[
                        bench["price_per_message_rub"]["source_url"],
                        bench["deliverability"]["source_url"],
                        bench["ctr"]["source_url"],
                    ],
                )
            )
            continue

        ctr_fallback = assumptions["marketplace_ctr"]["by_channel"].get(bench_key)
        ctr_lo, ctr_hi = _ctr_range(bench.get("ctr", {}), ctr_fallback)
        ctr_mid = (ctr_lo + ctr_hi) / 2
        ecpm_mid = _ecpm_mid(bench, ctr_mid) * jitter
        if bench.get("cpm_rub", {}).get("range"):
            # у источника есть явный диапазон, но он описывает категории целиком;
            # берём умеренную полуширину вокруг медианы, диапазон остаётся в note
            ecpm_range = _band(ecpm_mid, 0.30)
        else:
            ecpm_range = _band(ecpm_mid)
        cvr_lo, cvr_hi = range_of(assumptions["conversion_rate"]["by_kind"][kind])
        cap_mid = value_of(assumptions["audience_scale"]["daily_unique_audience_by_kind"][kind])
        sources = [
            node["source_url"]
            for key in ("cpm_rub", "cpc_rub", "ctr")
            if (node := bench.get(key)) and node.get("source_url")
        ]
        channels.append(
            CatalogChannel(
                channel_id=channel_id,
                family=family,
                display_name=bench["display_name"],
                expected_ecpm_range=ecpm_range,
                expected_ctr_range=(ctr_lo, ctr_hi),
                expected_cvr_range=(cvr_lo, cvr_hi),
                daily_unique_capacity_band=(int(cap_mid * 0.7), int(cap_mid * 1.3)),
                supports_video=bench_key == "social_video",
                benchmark_sources=sources,
            )
        )

    payload = json.dumps([c.model_dump(mode="json") for c in channels], sort_keys=True)
    catalog_id = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return PublicCatalog(catalog_id=catalog_id, channels=channels)
