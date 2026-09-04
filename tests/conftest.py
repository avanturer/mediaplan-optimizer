"""Общие фикстуры: каталог, ретро-история, кривые и план демо 1 считаются один раз на сессию."""

import pytest

from brain.curves import build_curves
from brain.planner import plan
from contracts import Brief
from harness.retro import collect_retro_history
from world import build_catalog


@pytest.fixture(scope="session")
def catalog():
    return build_catalog(0)


@pytest.fixture(scope="session")
def history(catalog):
    return collect_retro_history(catalog)


@pytest.fixture(scope="session")
def curves(history, catalog):
    return build_curves(history, catalog)


@pytest.fixture(scope="session")
def demo_brief(catalog):
    return Brief(budget_rub=1_200_000, horizon_days=21, channel_ids=catalog.channel_ids)


@pytest.fixture(scope="session")
def demo_plan(demo_brief, catalog, curves):
    return plan(demo_brief, catalog, curves)
