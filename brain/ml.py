"""CPU-модели из публичных ретро-наблюдений. Никаких импортов world.

Логистический классификатор изменений воронки, неотрицательная регрессия
вогнутых кривых отклика и регрессия попарных потерь уникального охвата.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np

from brain.assumptions import campaign_audience_multiplier
from brain.curves import CurvePoint, ResponseCurve
from contracts import ChannelObservation, PublicCatalog, RetroHistory
from contracts.ml import MLSignal

FEATURE_NAMES = ("ctr_change", "cvr_change", "cpm_change", "delivery_change",
                 "fraud_share", "fraud_change", "cap_change")


def nonnegative_fit(x: np.ndarray, y: np.ndarray, ridge: float = 0.001) -> np.ndarray:
    """Projected gradient NNLS с ridge, фиксированный воспроизводимый бюджет итераций."""
    scale = np.maximum(np.linalg.norm(x, axis=0), 1e-12)
    a = x / scale
    gram = a.T @ a + ridge * np.eye(a.shape[1])
    target = a.T @ y
    step = 1 / max(float(np.linalg.norm(gram, ord=2)), 1e-12)
    weights = np.zeros(a.shape[1])
    for _ in range(600):
        weights = np.maximum(0, weights - step * (gram @ weights - target))
    return weights / scale


def fit_response_curves(history: RetroHistory, curves: dict[str, ResponseCurve]) -> dict[str, ResponseCurve]:
    """Учится непосредственно на дневных outcomes, а не копирует функцию мира."""
    result = {}
    for cid, base in curves.items():
        samples = []
        for episode in history.episodes:
            # Для response-кривых нужны короткие пробы на свежей аудитории.
            if len(episode.observations) != 24:
                continue
            observations = [o.by_channel[cid] for o in episode.observations]
            samples.append([sum(o.spend for o in observations), sum(o.impressions for o in observations),
                            sum(o.clicks for o in observations), sum(o.conversions for o in observations)])
        if len(samples) < 4 or base.max_daily_spend <= 0:
            result[cid] = replace(base)
            continue
        data = np.array(samples, dtype=float)
        knots = np.geomspace(max(base.max_daily_spend * 0.01, 0.01), base.max_daily_spend, 12)
        x = np.minimum(data[:, 0, None] / knots, 1)
        grid = np.linspace(0, base.max_daily_spend, 64)
        design = np.minimum(grid[:, None] / knots, 1)
        fitted = [design @ nonnegative_fit(x, data[:, col]) for col in (1, 2, 3)]
        imps = fitted[0]
        clicks = np.minimum(fitted[1], imps)
        conv = np.minimum(fitted[2], clicks)
        points = [CurvePoint(float(b), float(i), float(c), float(v), float(i * base.reach_per_impression))
                  for b, i, c, v in zip(grid, imps, clicks, conv, strict=True)]
        result[cid] = replace(base, points=points, learned_rates=True, max_daily_impressions=float(imps[-1]))
    return result


@dataclass
class ReachModel:
    channel_ids: list[str]
    pools: dict[str, float]
    weights: np.ndarray

    def __post_init__(self):
        self.pairs = list(combinations(self.channel_ids, 2))

    def features(self, reached: dict[str, float]) -> np.ndarray:
        r = {cid: min(max(reached.get(cid, 0.0), 0), self.pools[cid]) for cid in self.channel_ids}
        return np.array([r[a] * r[b] / max(self.pools[a], self.pools[b], 1) for a, b in self.pairs])

    def predict(self, reached: dict[str, float]) -> float:
        if set(reached) - set(self.channel_ids):
            raise ValueError("модель охвата не обучена для этих каналов")
        values = [max(v, 0) for v in reached.values()]
        total = sum(values)
        return float(np.clip(total - self.features(reached) @ self.weights, max(values, default=0), total))

    def incremental(self, additions: dict[str, float], previous: dict[str, float] | None = None) -> float:
        previous = previous or {}
        combined = {cid: previous.get(cid, 0) + additions.get(cid, 0) for cid in self.channel_ids}
        return max(self.predict(combined) - self.predict(previous), 0.0)

    @classmethod
    def fit(cls, history: RetroHistory, catalog: PublicCatalog) -> ReachModel:
        pools = {ch.channel_id: float(ch.sms.base_size if ch.sms else ch.capacity_mid * campaign_audience_multiplier())
                 for ch in catalog.channels}
        model = cls(sorted(pools), pools, np.zeros(len(pools) * (len(pools) - 1) // 2))
        x, y = [], []
        for episode in history.episodes:
            reached = dict.fromkeys(model.channel_ids, 0.0)
            total = 0
            for index, obs in enumerate(episode.observations):
                if obs.deduplicated_reach is None:
                    raise ValueError("для обучения охвата нужны измерения deduplicated_reach")
                total += obs.total_reach
                for cid, row in obs.by_channel.items():
                    reached[cid] += row.unique_reach
                if (index + 1) % 6 == 0:
                    x.append(model.features(reached))
                    y.append(max(sum(reached.values()) - total, 0))
        if not x:
            raise ValueError("нет ретро-данных для обучения охвата")
        model.weights = nonnegative_fit(np.array(x), np.array(y))
        # Ограничение суммы весов гарантирует неотрицательный предельный охват.
        row_sums = {cid: sum(w for pair, w in zip(model.pairs, model.weights, strict=True) if cid in pair)
                    for cid in model.channel_ids}
        model.weights *= min(1, 0.9 / max(max(row_sums.values()), 1e-12))
        return model


@dataclass
class QualityWindow:
    rows: deque = field(default_factory=lambda: deque(maxlen=24))

    def update(self, obs: ChannelObservation, cap: float) -> np.ndarray | None:
        self.rows.append((obs, cap))
        if len(self.rows) < 24:
            return None
        rows = list(self.rows)
        old, recent = rows[:-6], rows[-6:]

        def aggregate(items):
            return np.array([sum(getattr(o, key) for o, _ in items) for key in
                             ("impressions", "clicks", "conversions", "spend", "requests")]
                            + [sum(o.fraud_share * o.impressions for o, _ in items), sum(c for _, c in items)], dtype=float)

        a, b = aggregate(old), aggregate(recent)
        if min(a[0], b[0]) < 200 or min(a[3], b[3]) <= 0:
            return None
        ctr_a = (a[1] + 1) / (a[0] + 100)
        cvr_a = (a[2] + 1) / (a[1] + 30)
        ctr_b = (b[1] + 200 * ctr_a) / (b[0] + 200)
        cvr_b = (b[2] + 20 * cvr_a) / (b[1] + 20)
        def ratio(v, ref):
            return float(np.clip(np.log(max(v, 1e-9) / max(ref, 1e-9)), -4, 4))
        fraud_a, fraud_b = a[5] / a[0], b[5] / b[0]
        return np.array([ratio(ctr_b, ctr_a), ratio(cvr_b, cvr_a), ratio(b[3] / b[0], a[3] / a[0]),
            ratio(b[0] / max(b[4], 1), a[0] / max(a[4], 1)), fraud_b, fraud_b - fraud_a,
            ratio(b[6] / 6, a[6] / 18)])


@dataclass
class QualityModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    threshold: float

    def score(self, x: np.ndarray) -> float:
        z = np.r_[1.0, (x - self.mean) / self.scale] @ self.weights
        return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, validation_x: np.ndarray, validation_y: np.ndarray) -> QualityModel:
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.01)
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        weights = np.zeros(design.shape[1])
        positive = max(float(y.mean()), 1e-3)
        sample_weights = np.where(y > 0, 0.5 / positive, 0.5 / max(1 - positive, 1e-3))
        for _ in range(800):
            p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
            penalty = 0.01 * weights
            penalty[0] = 0
            weights -= 0.05 * (design.T @ ((p - y) * sample_weights) / len(y) + penalty)
        model = cls(mean, scale, weights, 0.8)
        normal = [model.score(row) for row, label in zip(validation_x, validation_y, strict=True) if label == 0]
        model.threshold = max(0.65, float(np.quantile(normal, 0.99))) if normal else 0.9
        return model


@dataclass
class QualityMonitor:
    model: QualityModel
    windows: dict[str, QualityWindow] = field(default_factory=dict)
    streaks: dict[str, int] = field(default_factory=dict)

    def observe(self, cid: str, obs: ChannelObservation, cap: float) -> MLSignal:
        features = self.windows.setdefault(cid, QualityWindow()).update(obs, cap)
        if features is None:
            self.streaks[cid] = 0
            return MLSignal(threshold=self.model.threshold)
        score = self.model.score(features)
        self.streaks[cid] = self.streaks.get(cid, 0) + 1 if score >= self.model.threshold else 0
        alert = self.streaks[cid] >= 2
        reason = "воронка соответствует ретро-режиму"
        if score >= self.model.threshold:
            if features[4] > 0.05 and features[0] > 0:
                reason = "рост кликов вместе с подозрительным трафиком; риск фрода"
            elif features[1] < 0:
                reason = "конверсия из кликов снизилась"
            elif features[2] > 0:
                reason = "стоимость показов выросла"
            else:
                reason = "изменилась отдача канала"
        return MLSignal(score=score, threshold=self.model.threshold, alert=alert, reason=reason)


@dataclass
class MLBundle:
    catalog_id: str
    curves: dict[str, ResponseCurve]
    reach: ReachModel
    quality: QualityModel
    training_summary: dict
    model_id: str = ""

    def __post_init__(self):
        if not self.model_id:
            content = self.to_dict()
            self.model_id = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"catalog_id": self.catalog_id, "training_summary": self.training_summary,
                "quality": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self.quality).items()},
                "reach": {"channel_ids": self.reach.channel_ids, "pools": self.reach.pools, "weights": self.reach.weights.tolist()},
                "curves": {cid: {**vars(c), "points": [vars(p) for p in c.points], "hourly_profile": c.hourly_profile.tolist()} for cid, c in self.curves.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> MLBundle:
        q = data["quality"]
        quality = QualityModel(np.array(q["mean"]), np.array(q["scale"]), np.array(q["weights"]), q["threshold"])
        r = data["reach"]
        reach = ReachModel(r["channel_ids"], r["pools"], np.array(r["weights"]))
        curves = {cid: ResponseCurve(**{**c, "points": [CurvePoint(**p) for p in c["points"]],
                                       "hourly_profile": np.array(c["hourly_profile"])}) for cid, c in data["curves"].items()}
        return cls(data["catalog_id"], curves, reach, quality, data["training_summary"])
