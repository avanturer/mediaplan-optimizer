"""Парное сравнение стратегий на общей ленте случайных событий.

Одинаковые ``world_seed`` и ``noise_seed`` для всех стратегий, меняется
только policy (контракт мира, §9 и §13). Один красивый seed не считается
доказательством: сравнение идёт минимум на 20–30 парных прогонах, отчёт
содержит среднее, разброс и парную дельту относительно static.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain.curves import ResponseCurve
from brain.ml import MLBundle
from contracts import MediaPlan, PublicCatalog, RunSummary, SeedBundle, ShockEvent
from contracts.ml import MLConfig
from harness.runner import RunConfig, run_campaign
from world.settings import WorldSettings
from world.simulator import Simulator

METRICS = ("mape_spend", "mape_kpi", "wape_spend", "wape_kpi", "final_deviation_spend", "final_deviation_kpi", "unsmoothness", "lambda_cv")


@dataclass
class StrategyStats:
    strategy: str
    runs: list[RunSummary]
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    ci95: dict[str, float] = field(default_factory=dict)
    paired_delta_vs_static: dict[str, float] = field(default_factory=dict)
    win_rate_vs_static: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n": len(self.runs),
            "mean": self.mean,
            "std": self.std,
            "ci95": self.ci95,
            "paired_delta_vs_static": self.paired_delta_vs_static,
            "win_rate_vs_static": self.win_rate_vs_static,
            "mean_actual_kpi": float(np.mean([r.actual_kpi for r in self.runs])),
            "mean_actual_spend": float(np.mean([r.actual_spend for r in self.runs])),
        }


def compare_strategies(
    plan: MediaPlan,
    catalog: PublicCatalog,
    curves: dict[str, ResponseCurve],
    strategies: tuple[str, ...] = ("static", "proportional_pacing", "pid", "adaptive"),
    scenario_id: str = "stable",
    seeds: int = 20,
    injected: list[ShockEvent] | None = None,
    catalog_seed: int = 0,
    first_seed: int = 1,
    world_settings: WorldSettings | None = None,
    ml: MLConfig | None = None,
    ml_bundle: MLBundle | None = None,
    hold_plan: bool = True,
) -> dict[str, StrategyStats]:
    sim = Simulator(catalog, settings=world_settings)
    results: dict[str, list[RunSummary]] = {s: [] for s in strategies}
    for k in range(first_seed, first_seed + seeds):
        bundle = SeedBundle(catalog_seed=catalog_seed, world_seed=k, noise_seed=10_000 + k)
        for strategy in strategies:
            config = RunConfig(strategy=strategy, scenario_id=scenario_id, seeds=bundle, injected=list(injected or []),
                               ml=ml if strategy != "static" and ml else MLConfig(), hold_plan=hold_plan)
            results[strategy].append(run_campaign(plan, catalog, curves, config, simulator=sim, ml_bundle=ml_bundle))

    stats: dict[str, StrategyStats] = {}
    base = results.get("static")
    for strategy, runs in results.items():
        st = StrategyStats(strategy=strategy, runs=runs)
        for metric in METRICS:
            values = np.array([getattr(r, metric) for r in runs])
            st.mean[metric] = float(values.mean())
            st.std[metric] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            st.ci95[metric] = float(1.96 * st.std[metric] / np.sqrt(len(values))) if len(values) > 1 else 0.0
            if base is not None and strategy != "static":
                base_values = np.array([getattr(r, metric) for r in base])
                delta = values - base_values
                st.paired_delta_vs_static[metric] = float(delta.mean())
                st.win_rate_vs_static[metric] = float(np.mean(values < base_values))
        stats[strategy] = st
    return stats


def summary_table(stats: dict[str, StrategyStats]) -> str:
    header = f"{'strategy':22s} {'MAPE spend':>11s} {'MAPE kpi':>9s} {'dev spend':>10s} {'dev kpi':>8s} {'unsmooth':>9s} {'λ cv':>6s}"
    rows = [header]
    for name, st in stats.items():
        rows.append(
            f"{name:22s} {st.mean['mape_spend']:>10.1%} {st.mean['mape_kpi']:>8.1%} "
            f"{st.mean['final_deviation_spend']:>9.1%} {st.mean['final_deviation_kpi']:>7.1%} "
            f"{st.mean['unsmoothness']:>8.2f} {st.mean['lambda_cv']:>6.2f}"
        )
    return "\n".join(rows)
