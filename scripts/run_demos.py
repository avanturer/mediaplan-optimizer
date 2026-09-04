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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
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
    for scenario in ("stable", "ctr_drop", "cpm_spike", "channel_pause", "capacity_cut"):
        stats = compare_strategies(demo1, catalog, curves, scenario_id=scenario, seeds=args.seeds)
        comparison[scenario] = {name: st.to_dict() for name, st in stats.items()}
        report += [f"### Сценарий {scenario}", "", "```", summary_table(stats), "```", ""]
        adaptive_st, static_st = stats["adaptive"], stats["static"]
        report.append(
            f"Победы adaptive над static по отклонению KPI в конце: {adaptive_st.win_rate_vs_static['final_deviation_kpi']:.0%} миров; "
            f"парная дельта MAPE расхода {adaptive_st.paired_delta_vs_static['mape_spend']:+.1%}, "
            f"KPI в среднем {adaptive_st.to_dict()['mean_actual_kpi']:,.0f} против {static_st.to_dict()['mean_actual_kpi']:,.0f}."
        )
        report.append("")
    (RESULTS / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    report.append(f"Время стенда: {time.perf_counter() - started:.0f} с.")
    (RESULTS / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
