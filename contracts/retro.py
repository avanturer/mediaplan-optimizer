"""Ретро-история: публичные наблюдения прошлых кампаний в том же мире.

Презентация кейса разрешает планировщику «кривые, посчитанные на ретро данных
модели мира». Harness прогоняет пробные кампании через публичный reset/step на
отдельном ``world_seed`` и складывает сюда только действия и наблюдения, то есть
ровно то, что видел бы рекламодатель в кабинете. Скрытых параметров здесь нет
по построению, поэтому утечка невозможна даже по неосторожности.
"""

from pydantic import BaseModel, Field

from contracts.simulation import Action, Observation


class RetroEpisode(BaseModel):
    episode_id: str
    horizon_hours: int = Field(ge=1)
    actions: list[Action]
    observations: list[Observation]


class RetroHistory(BaseModel):
    catalog_id: str
    episodes: list[RetroEpisode] = Field(default_factory=list)

    @property
    def total_hours(self) -> int:
        return sum(len(e.observations) for e in self.episodes)
