"""Модель мира и симулятор рынка.

Владеет истинными параметрами каналов, скрытым состоянием и сценарием шоков.
Наружу отдаёт только публичный каталог (``build_catalog``) и часовые
наблюдения через ``Simulator.reset / step``. Код планировщика и исполнителя
этот пакет не импортирует (проверяется import-linter).
"""

from world.catalog_builder import build_catalog
from world.scenarios import SCENARIOS, get_scenario
from world.simulator import Simulator

__all__ = ["SCENARIOS", "Simulator", "build_catalog", "get_scenario"]
