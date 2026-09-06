"""Reproducible holdout evaluation; never tunes models on these seeds."""

import json
from pathlib import Path

import numpy as np

from brain.curves import build_curves
from brain.planner import plan
from contracts import Brief, SeedBundle
from contracts.ml import MLConfig
from harness.ml_training import collect_ml_history, train_ml_bundle
from harness.retro import collect_retro_history
from harness.runner import RunConfig, run_campaign
from world import build_catalog


def stats(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)))


def main():
    catalog = build_catalog(0)
    history = collect_retro_history(catalog)
    curves = build_curves(history, catalog)
    bundle = train_ml_bundle(catalog, curves, history)
    report = {"model_id": bundle.model_id, "training": bundle.training_summary,
              "evaluation_seeds": list(range(1, 31)), "scenarios": {}}
    heldout, x, y = collect_ml_history(catalog, tuple(range(3000, 3012)))
    predicted = np.array([bundle.quality.score(row) >= bundle.quality.threshold for row in x])
    report["quality_holdout"] = dict(precision=float(np.sum(predicted & (y == 1))/max(predicted.sum(), 1)),
        recall=float(np.sum(predicted & (y == 1))/max(y.sum(), 1)),
        false_positive_rate=float(np.sum(predicted & (y == 0))/max(np.sum(y == 0), 1)),
        threshold=bundle.quality.threshold, rows=len(y), note="Window-level metrics before two-hour confirmation; synthetic labels")
    reach_errors, additive_errors = [], []
    for episode in heldout.episodes:
        local = {cid: sum(o.by_channel[cid].unique_reach for o in episode.observations) for cid in catalog.channel_ids}
        actual = sum(o.total_reach for o in episode.observations)
        reach_errors.append(abs(bundle.reach.predict(local)-actual)/max(actual, 1))
        additive_errors.append(abs(sum(local.values())-actual)/max(actual, 1))
    report["reach_holdout_relative_error"] = {"ml": stats(reach_errors), "additive": stats(additive_errors)}
    response_holdout = collect_retro_history(catalog, world_seeds=(3100, 3101, 3102))
    errors = {"ml": [], "baseline": []}
    actual_sum = 0
    for episode in response_holdout.episodes:
        if len(episode.observations) != 24:
            continue
        for cid in catalog.channel_ids:
            spend = sum(o.by_channel[cid].spend for o in episode.observations)
            actual = sum(o.by_channel[cid].conversions for o in episode.observations)
            actual_sum += actual
            for name, cs in (("ml", bundle.curves), ("baseline", curves)):
                ctr, cvr = cs[cid].rates_at(spend)
                errors[name].append(abs(cs[cid].impressions_at(spend)*ctr*cvr-actual))
    report["response_holdout_conversions_wape"] = {key: sum(values)/max(actual_sum, 1) for key, values in errors.items()}
    media_plan = plan(Brief(budget_rub=1_200_000, horizon_days=21, channel_ids=catalog.channel_ids), catalog, curves)
    on = MLConfig(anomaly_detection=True, response_curves=True, reach_correction=True)
    for scenario in ("stable", "fraud_surge"):
        summaries = {name: [] for name in ("static", "adaptive_without_ml", "adaptive_ml")}
        for seed in range(1, 31):
            for name in summaries:
                run = run_campaign(media_plan, catalog, curves, RunConfig(
                    strategy="static" if name == "static" else "adaptive", scenario_id=scenario,
                    seeds=SeedBundle(world_seed=seed, noise_seed=10000+seed),
                    ml=on if name == "adaptive_ml" else MLConfig()), ml_bundle=bundle)
                post = run.hours[240:]
                post_errors = [abs(h.fact_cum_kpi-h.plan_cum_kpi) for h in post]
                violations = int(run.actual_spend > media_plan.total_budget_rub+1e-6)
                for hour in run.hours:
                    for cid, ch in hour.fact_by_channel.items():
                        violations += int(not (0 <= ch["spend"] <= hour.caps[cid]+1e-6 and
                            0 <= ch["unique_reach"] <= ch["impressions"] <= ch["requests"] and
                            0 <= ch["conversions"] <= ch["clicks"] <= ch["impressions"]))
                summaries[name].append(dict(seed=seed, actual_kpi=run.actual_kpi, spend=run.actual_spend,
                    wape_kpi=run.wape_kpi, mape_kpi=run.mape_kpi, final_deviation_kpi=run.final_deviation_kpi,
                    final_absolute_error=abs(run.actual_kpi-run.promised_kpi),
                    post_shock_wape=sum(post_errors)/sum(h.plan_cum_kpi for h in post),
                    post_shock_mae=float(np.mean(post_errors)), constraint_violations=violations))
            print(scenario, seed, flush=True)
        metrics = {}
        for name, rows in summaries.items():
            metrics[name] = {key: stats([row[key] for row in rows]) for key in rows[0] if key not in ("seed", "constraint_violations")}
            metrics[name]["constraint_violations"] = sum(row["constraint_violations"] for row in rows)
        delta = [a["wape_kpi"]-b["wape_kpi"] for a, b in zip(summaries["adaptive_ml"], summaries["adaptive_without_ml"], strict=True)]
        report["scenarios"][scenario] = dict(metrics=metrics, paired_wape_delta=stats(delta), runs=summaries)
    Path("results/ml_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    Path("results/ml_model.json").write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2)+"\n")
    lines = ["# ML: независимая проверка", "", f"Модель `{bundle.model_id}`. 30 paired seeds × 2 сценария × 3 стратегии = 180 эпизодов по 21 дню.",
             "Один базовый утверждённый план; оценивается ML исполнения, а не эффект другого плана. Настройки не подбирались на тесте.", "",
             "| Сценарий / стратегия | WAPE KPI, среднее ± σ | Финальное отклонение | WAPE после часа 240 | Нарушения |",
             "|---|---:|---:|---:|---:|"]
    for scenario, section in report["scenarios"].items():
        for name, row in section["metrics"].items():
            lines.append(f"| {scenario} / {name} | {row['wape_kpi']['mean']:.2%} ± {row['wape_kpi']['std']:.2%} | {row['final_deviation_kpi']['mean']:.2%} | {row['post_shock_wape']['mean']:.2%} | {row['constraint_violations']} |")
    lines += ["", "Снижение WAPE — лучше. В JSON сохранены разброс, MAPE, абсолютные ошибки, расход, парные разности и результаты каждого seed.",
              "", "## Проверки самих моделей", "", "```json", json.dumps({k:v for k,v in report.items() if 'holdout' in k}, ensure_ascii=False, indent=2), "```",
              "", "Все данные синтетические. Оценки не доказывают переносимость на реальные рекламные кабинеты; score детектора не калиброван как вероятность фрода."]
    Path("results/ml_validation.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
