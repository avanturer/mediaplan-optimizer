"""Анти-утечка: мозг не видит мира.

Структурно проверяется import-linter (pyproject.toml). Здесь поведенческая
проверка: план зависит только от каталога и ретро-истории, а смена зерна
оцениваемого мира его не меняет.
"""

import ast
from pathlib import Path

from brain.planner import plan
from contracts import Brief, SeedBundle
from harness.runner import RunConfig, run_campaign

ROOT = Path(__file__).resolve().parent.parent


def test_plan_independent_of_world_seed(catalog, curves, demo_brief):
    """Один и тот же бриф и каталог: план побайтово одинаков, какой бы мир его потом ни исполнял."""
    plan_a = plan(demo_brief, catalog, curves)
    plan_b = plan(demo_brief, catalog, curves)
    assert plan_a.model_dump() == plan_b.model_dump()
    # два разных мира исполняют один план: факт различается, план тот же
    run_a = run_campaign(plan_a, catalog, curves, RunConfig(seeds=SeedBundle(world_seed=1, noise_seed=1)))
    run_b = run_campaign(plan_a, catalog, curves, RunConfig(seeds=SeedBundle(world_seed=2, noise_seed=2)))
    assert run_a.promised_kpi == run_b.promised_kpi
    assert run_a.actual_kpi != run_b.actual_kpi


def test_brain_never_imports_world():
    offenders = []
    for path in (ROOT / "brain").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "world" or n.startswith("world.") for n in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"brain импортирует world: {offenders}"


def test_retro_history_contains_only_public_fields(history):
    payload = history.model_dump_json()
    for forbidden in ("latent", "ecpm_base", "base_ctr", "fatigue", "world_seed"):
        assert forbidden not in payload


def test_type_b_plan_also_independent_of_world(catalog, curves):
    from contracts import TargetKpi

    brief = Brief(target_kpi=TargetKpi.CLICKS, target_value=30_000, horizon_days=14, channel_ids=catalog.channel_ids)
    assert plan(brief, catalog, curves).model_dump() == plan(brief, catalog, curves).model_dump()
