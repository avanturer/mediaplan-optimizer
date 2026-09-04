"""Контракты границ между модулями MediaPlan Optimizer.

Три пакета общаются только через эти схемы:

- ``world`` (модель мира и симулятор) отдаёт наружу публичный каталог,
  часовые наблюдения и принимает действие «лимиты расхода на час»;
- ``brain`` (планировщик и сервис исполнения) видит только каталог,
  ретро-наблюдения и накопленный факт;
- ``app`` (веб-кабинет) вызывает оба через harness.

Правило: ``brain`` никогда не импортирует ``world``. Проверяется import-linter
и поведенческим тестом: смена ``world_seed`` не меняет план.
"""

from contracts.brief import Brief, Objective, TargetKpi
from contracts.catalog import CatalogChannel, ChannelFamily, PublicCatalog, SmsPublic
from contracts.execution import (
    ChannelDecision,
    ChannelStatus,
    ExecutionDecision,
    HourRecord,
    Proposal,
    RunSummary,
    TrackingStatus,
)
from contracts.plan import (
    BindingConstraint,
    BriefSuggestion,
    CalendarCell,
    ChannelAllocation,
    Forecast,
    Infeasibility,
    MediaPlan,
    TrajectoryPoint,
)
from contracts.retro import RetroEpisode, RetroHistory
from contracts.simulation import (
    Action,
    ChannelObservation,
    Observation,
    Scenario,
    SeedBundle,
    ShockEvent,
    ShockParameter,
    StepInfo,
    StepMetrics,
)

__all__ = [
    "Action",
    "BindingConstraint",
    "Brief",
    "BriefSuggestion",
    "CalendarCell",
    "CatalogChannel",
    "ChannelAllocation",
    "ChannelDecision",
    "ChannelFamily",
    "ChannelObservation",
    "ChannelStatus",
    "ExecutionDecision",
    "Forecast",
    "HourRecord",
    "Infeasibility",
    "MediaPlan",
    "Objective",
    "Observation",
    "Proposal",
    "PublicCatalog",
    "RetroEpisode",
    "RetroHistory",
    "RunSummary",
    "Scenario",
    "SeedBundle",
    "ShockEvent",
    "ShockParameter",
    "SmsPublic",
    "StepInfo",
    "StepMetrics",
    "TargetKpi",
    "TrackingStatus",
    "TrajectoryPoint",
]
