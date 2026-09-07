"""30 парных миров: регрессионный отчёт, без подбора настроек по результату."""

import argparse
import json
from pathlib import Path

import numpy as np

from brain.curves import build_curves
from brain.planner import plan
from contracts import Brief
from harness.compare import compare_strategies
from harness.retro import collect_retro_history
from world import build_catalog
from world.settings import Competitor, WorldSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/world_extensions_validation.json"))
    args = parser.parse_args()
    catalog = build_catalog()
    curves = build_curves(collect_retro_history(catalog), catalog)
    media_plan = plan(Brief(budget_rub=1_200_000, horizon_days=21, channel_ids=catalog.channel_ids), catalog, curves)
    report = {}
    scenes = [("stable", 240, None), ("fraud_surge", 240, None), ("sms_weekly_limit", 168, None),
              ("competition", 240, WorldSettings(competitors=[Competitor(competitor_id="rival",
                  strength=0.8, channel_advantages={"programmatic": 2, "marketplace_1": 1}, start_hour=240)]))]
    for name, shock_hour, settings in scenes:
        stats = compare_strategies(media_plan, catalog, curves, strategies=("static", "adaptive"),
            scenario_id="stable" if name == "competition" else name, seeds=args.seeds, world_settings=settings)
        report[name] = {}
        for strategy, st in stats.items():
            data = st.to_dict()
            post_wape, post_abs, violations = [], [], 0
            for run in st.runs:
                rows = run.hours[shock_hour:]
                actual = np.array([h.fact_cum_kpi for h in rows])
                promised = np.array([h.plan_cum_kpi for h in rows])
                post_abs.append(float(np.abs(actual - promised).mean()))
                post_wape.append(float(np.abs(actual - promised).sum() / promised.sum()))
                violations += int(run.actual_spend > media_plan.total_budget_rub + 1e-6)
                for h in run.hours:
                    for cid, ch in h.fact_by_channel.items():
                        violations += int(not (0 <= ch["spend"] <= h.caps[cid] + 1e-6
                            and 0 <= ch["unique_reach"] <= ch["impressions"] <= ch["requests"]
                            and 0 <= ch["conversions"] <= ch["clicks"] <= ch["impressions"]))
            data["post_shock_cumulative_kpi_wape"] = {"mean": float(np.mean(post_wape)), "std": float(np.std(post_wape, ddof=1))}
            data["post_shock_cumulative_kpi_mae"] = {"mean": float(np.mean(post_abs)), "std": float(np.std(post_abs, ddof=1))}
            data["constraint_violations"] = violations
            report[name][strategy] = data
        print(name, {s: round(d["mean"]["final_deviation_kpi"], 4) for s, d in report[name].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
