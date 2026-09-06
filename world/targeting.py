"""Публичный каталог под выбранный сегмент, без знания оцениваемого мира."""

import hashlib
import json

import yaml

from contracts.catalog import PublicCatalog
from contracts.targeting import AudienceTargeting
from world.config import CONFIG_DIR


def catalog_for_targeting(catalog: PublicCatalog, targeting: AudienceTargeting) -> PublicCatalog:
    if catalog.targeting == targeting:
        return catalog
    if catalog.targeting != AudienceTargeting():
        raise ValueError("для смены сегмента требуется исходный каталог без таргетинга")
    with (CONFIG_DIR / "world_extensions.yaml").open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh)["targeting"]
    share = 1.0
    for axis in ("age_groups", "genders", "geo"):
        selected = getattr(targeting, axis) or list(config[axis])
        share *= sum(config[axis][key] for key in selected)
    regions = targeting.geo or list(config["geo"])
    geo_price = sum(config["geo"][g] * config["geo_price"][g] for g in regions)
    geo_price /= sum(config["geo"][g] for g in regions)
    broad_price = sum(config["geo"][g] * config["geo_price"][g] for g in config["geo"])
    price = min(config["max_price_multiplier"], geo_price / broad_price * share ** -config["scarcity_elasticity"])
    result = catalog.model_copy(deep=True)
    result.targeting = targeting.model_copy(deep=True)
    for ch in result.channels:
        ch.daily_unique_capacity_band = tuple(max(1, int(x * share)) for x in ch.daily_unique_capacity_band)
        ch.expected_ecpm_range = tuple(x * price for x in ch.expected_ecpm_range)
        if ch.sms is not None:
            ch.sms.base_size = max(1, int(ch.sms.base_size * share))
            ch.sms.price_per_message_rub *= price
    payload = result.model_dump(mode="json", exclude={"catalog_id"})
    result.catalog_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return result
