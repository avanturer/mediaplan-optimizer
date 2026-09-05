"""Общие агрегированные пулы пар каналов, без идентификаторов пользователей."""

import math
from dataclasses import dataclass
from itertools import combinations

from world.settings import WorldSettings


@dataclass
class Cohort:
    channels: tuple[str, ...]
    size: float
    reached: float = 0.0


class Audience:
    def __init__(self, pools: dict[str, float], settings: WorldSettings):
        self.pools = pools
        self.cohorts: list[Cohort] = []
        matrix = settings.overlap_matrix
        ids = sorted(pools)
        if matrix is not None:
            if set(matrix) != set(ids) or any(set(row) != set(ids) for row in matrix.values()):
                raise ValueError("матрица пересечений должна содержать все каналы каталога")
            for a in ids:
                for b in ids:
                    x = matrix[a][b]
                    if not math.isfinite(x) or not 0 <= x <= 1:
                        raise ValueError("пересечения должны быть конечными долями [0, 1]")
                    if x != matrix[b][a] or (a == b and x != 1):
                        raise ValueError("матрица должна быть симметричной с единицами на диагонали")
        shared = dict.fromkeys(ids, 0.0)
        for a, b in combinations(ids, 2):
            overlap = matrix[a][b] if matrix is not None else settings.default_overlap
            size = overlap * min(pools[a], pools[b])
            shared[a] += size
            shared[b] += size
            if size:
                self.cohorts.append(Cohort((a, b), size))
        for cid in ids:
            remaining = pools[cid] - shared[cid]
            if remaining < -1e-8:
                raise ValueError(f"{cid}: сумма попарных общих пулов превышает аудиторию; тройные пересечения не поддерживаются")
            if remaining > 0:
                self.cohorts.append(Cohort((cid,), remaining))

    def step(self, human_impressions: dict[str, int]) -> int:
        before = sum(c.reached for c in self.cohorts)
        for cohort in self.cohorts:
            hazard = sum(human_impressions.get(cid, 0) / self.pools[cid] for cid in cohort.channels)
            cohort.reached += (cohort.size - cohort.reached) * -math.expm1(-hazard)
        after = sum(c.reached for c in self.cohorts)
        return max(0, round(after) - round(before))
