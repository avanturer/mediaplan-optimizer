"""Три демонстрации кейса и стенд сравнения стратегий.

Запуск: ``python scripts/run_demos.py --seeds 20``. Пишет в ``results/``:
``demo1_plan.json``, ``demo2_verdict.json``, ``demo3_run.json`` и
``comparison.json`` плюс ``report.md`` с таблицами, которые попадают в README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.curves import build_curves  # noqa: E402
from brain.planner import plan  # noqa: E402
from contracts import Brief, SeedBundle, ShockEvent, ShockParameter, TargetKpi  # noqa: E402
from harness.compare import compare_strategies, summary_table  # noqa: E402
from harness.retro import collect_retro_history  # noqa: E402
from harness.runner import RunConfig, run_campaign  # noqa: E402
from world import build_catalog  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
NARROW_PRESET = ["social_2", "social_3", "marketplace_1", "sms"]


def _histogram(dev_a, dev_s, scenario: str, seeds: int, out_dir: Path, threshold: float) -> None:
    """Гистограмма отклонений KPI по мирам: наша стратегия против заморозки. Без matplotlib молча пропускается."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = np.arange(0, max(float(dev_a.max()), float(dev_s.max()), threshold) + 0.05, 0.025)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.hist(dev_s * 1, bins=bins, alpha=0.55, color="#C9902E", label="заморозка")
    ax.hist(dev_a * 1, bins=bins, alpha=0.7, color="#1A9D78", label="наша стратегия")
    ax.axvline(threshold, color="#B84A3A", ls="--", lw=1.2)
    ax.text(threshold + 0.005, ax.get_ylim()[1] * 0.9, "порог кейса 20 %", color="#B84A3A")
    ax.set_xlabel("отклонение KPI в конце кампании")
    ax.set_ylabel("миров")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{seeds} парных миров, сценарий {scenario}: медианы {np.median(dev_a):.1%} против {np.median(dev_s):.1%}", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"stand_hist_{scenario}.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--scenarios", default="stable,ctr_drop,cpm_spike,channel_pause,capacity_cut")
    parser.add_argument("--figures", default="docs/figures", help="куда писать гистограммы отклонений (нужен matplotlib)")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    started = time.perf_counter()
    catalog = build_catalog(0)
    history = collect_retro_history(catalog)
    curves = build_curves(history, catalog)
    report: list[str] = ["# Результаты стенда", ""]

    # --- Демо 1: 1,2 млн ₽, 21 день, максимум конверсий
    demo1 = plan(Brief(budget_rub=1_200_000, horizon_days=21, channel_ids=catalog.channel_ids), catalog, curves)
    (RESULTS / "demo1_plan.json").write_text(demo1.model_dump_json(indent=2), encoding="utf-8")
    report += ["## Демо 1. Бюджет 1,2 млн ₽, 21 день, максимум конверсий", ""]
    report += [f"Прогноз: {demo1.total_kpi:,.0f} конверсий (P10–P90: {demo1.forecast.p10:,.0f}–{demo1.forecast.p90:,.0f}), CPA {demo1.total_budget_rub / demo1.total_kpi:,.0f} ₽", ""]
    report += ["| Канал | Бюджет, ₽ | Доля | Конверсии | CPA, ₽ | CPM, ₽ | Частота | Ёмкость | Цена след. 1000, тыс. ₽ |", "|---|---|---|---|---|---|---|---|---|"]
    for a in demo1.allocations:
        report.append(
            f"| {a.channel_id} | {a.budget_rub:,.0f} | {a.budget_rub / demo1.total_budget_rub:.0%} | {a.conversions:,.0f} | "
            f"{(a.cpa_rub or 0):,.0f} | {a.cpm_rub:,.0f} | {a.frequency:.1f} | {a.capacity_utilization:.0%} | "
            f"{(a.marginal_cost_per_1000_kpi_rub or 0) / 1000:,.0f} |"
        )
    report.append("")

    # --- Демо 2: 50 000 кликов за 14 дней, два пресета
    demo2_all = plan(Brief(target_kpi=TargetKpi.CLICKS, target_value=50_000, horizon_days=14, channel_ids=catalog.channel_ids), catalog, curves)
    demo2_narrow = plan(Brief(target_kpi=TargetKpi.CLICKS, target_value=50_000, horizon_days=14, channel_ids=NARROW_PRESET), catalog, curves)
    (RESULTS / "demo2_verdict.json").write_text(
        json.dumps({"all_channels": demo2_all.model_dump(mode="json"), "narrow_preset": demo2_narrow.model_dump(mode="json")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report += ["## Демо 2. 50 000 кликов за 14 дней", ""]
    report.append(f"Все восемь каналов: достижимо, достаточный бюджет {demo2_all.total_budget_rub:,.0f} ₽, прогноз {demo2_all.total_kpi:,.0f} кликов, шанс выполнить цель {demo2_all.forecast.probability_of_target:.0%}.")
    d = demo2_narrow.infeasibility
    report.append(f"Узкий пресет ({', '.join(NARROW_PRESET)}): недостижимо. {d.explanation} Связывает: {d.binding_constraint.value}. Ходы:")
    for s in d.suggestions:
        report.append(f"- {s.description}: {s.expected_kpi:,.0f} кликов при бюджете {s.expected_budget_rub:,.0f} ₽")
    report.append("")

    # --- Демо 3: шок из интерфейса и сравнение с заморозкой
    shock = ShockEvent(start_hour=240, target_channels=["marketplace_1"], parameter=ShockParameter.CTR, multiplier=0.6)
    seeds = SeedBundle(world_seed=1, noise_seed=10001)
    adaptive = run_campaign(demo1, catalog, curves, RunConfig("adaptive", "stable", seeds, [shock]))
    frozen = run_campaign(demo1, catalog, curves, RunConfig("static", "stable", seeds, [shock]))
    (RESULTS / "demo3_run.json").write_text(
        json.dumps({"adaptive": adaptive.model_dump(mode="json"), "frozen": frozen.model_dump(mode="json")}, ensure_ascii=False),
        encoding="utf-8",
    )
    report += ["## Демо 3. План демо 1, CTR −40 % в marketplace_1 с 240-го часа (шок из интерфейса)", ""]
    report += ["| Стратегия | Расход | Конверсии | MAPE расхода | MAPE KPI | Отклонение в конце (KPI) | Детектор |", "|---|---|---|---|---|---|---|"]
    for r in (adaptive, frozen):
        report.append(
            f"| {r.strategy} | {r.actual_spend:,.0f} / {r.promised_spend:,.0f} | {r.actual_kpi:,.0f} / {r.promised_kpi:,.0f} | "
            f"{r.mape_spend:.1%} | {r.mape_kpi:.1%} | {r.final_deviation_kpi:.1%} | {r.detection_hours or '—'} |"
        )
    report.append("")

    # --- Стенд: четыре стратегии, сценарии, парные миры
    comparison: dict[str, dict] = {}
    report += [f"## Стенд: {args.seeds} парных миров, четыре стратегии", ""]
    threshold = 0.20  # порог кейса: отклонение в конце не более 20 %
    for scenario in args.scenarios.split(","):
        stats = compare_strategies(demo1, catalog, curves, scenario_id=scenario, seeds=args.seeds)
        comparison[scenario] = {name: st.to_dict() for name, st in stats.items()}
        report += [f"### Сценарий {scenario}", "", "```", summary_table(stats), "```", ""]
        adaptive_st, static_st = stats["adaptive"], stats["static"]
        dev_a = np.array([r.final_deviation_kpi for r in adaptive_st.runs])
        dev_s = np.array([r.final_deviation_kpi for r in static_st.runs])
        report.append(
            f"Победы adaptive над static по отклонению KPI в конце: {adaptive_st.win_rate_vs_static['final_deviation_kpi']:.0%} миров; "
            f"парная дельта MAPE расхода {adaptive_st.paired_delta_vs_static['mape_spend']:+.1%}, "
            f"KPI в среднем {adaptive_st.to_dict()['mean_actual_kpi']:,.0f} против {static_st.to_dict()['mean_actual_kpi']:,.0f}."
        )
        report.append(
            f"Распределение отклонения KPI в конце по {args.seeds} мирам: наша медиана {np.median(dev_a):.1%}, "
            f"P90 {np.percentile(dev_a, 90):.1%}, в пороге кейса {np.mean(dev_a <= threshold):.0%} миров; "
            f"заморозка: медиана {np.median(dev_s):.1%}, P90 {np.percentile(dev_s, 90):.1%}, в пороге {np.mean(dev_s <= threshold):.0%}."
        )
        alarms = sum(len(r.detection_hours) for r in stats["static"].runs) / max(len(stats["static"].runs), 1)
        report.append(f"Срабатываний детектора на кампанию (static, без учёта причины): {alarms:.2f}.")
        report.append("")
        _histogram(dev_a, dev_s, scenario, args.seeds, Path(args.figures), threshold)
    (RESULTS / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    report.append(f"Время стенда: {time.perf_counter() - started:.0f} с.")
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
