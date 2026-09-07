"""Минимальный агрегированный таргетинг; названия гео обозначают синтетические группы."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AgeGroup = Literal["18_24", "25_34", "35_44", "45_54", "55_plus"]
Gender = Literal["female", "male"]
GeoGroup = Literal["capital", "large_cities", "other_regions"]


class AudienceTargeting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_groups: list[AgeGroup] = Field(default_factory=list)
    genders: list[Gender] = Field(default_factory=list)
    geo: list[GeoGroup] = Field(default_factory=list)

    @field_validator("age_groups", "genders", "geo")
    @classmethod
    def canonical_selection(cls, values: list[str]) -> list[str]:
        # Пустой список = вся аудитория; внутри измерения OR, между измерениями AND.
        return sorted(set(values))
