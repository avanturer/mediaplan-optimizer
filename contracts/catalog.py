"""Публичный каталог каналов: единственное, что планировщик знает о рынке заранее.

Схема согласована с контрактом модели мира (docs/world/WORLD_MODEL.md, §11):
каталог содержит ожидания в виде диапазонов, а не истинные параметры эпизода.
Истинные параметры мира выбираются внутри диапазонов или рядом с ними и
никогда сюда не попадают. Кривые «бюджет → отклик» планировщик строит сам из
ретро-наблюдений (contracts/retro.py), как это делает Google Reach Planner
по истории похожих кампаний.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ChannelFamily(StrEnum):
    """Три семейства поведения вместо восьми копий логики (контракт мира, §4)."""

    AUCTION = "auction"
    MARKETPLACE = "marketplace"
    DIRECT = "direct"


class SmsPublic(BaseModel):
    """Публичные параметры SMS: цена фиксированная, база конечная, частота ограничена."""

    price_per_message_rub: float = Field(gt=0)
    deliverability: float = Field(gt=0, le=1)
    base_size: int = Field(gt=0, description="размер абонентской базы сегмента")
    cooldown_days: int = Field(ge=1, description="не чаще одного сообщения абоненту раз в N дней")
    send_hours: tuple[int, int] = Field(
        default=(9, 21), description="окно отправки по местному времени, как у МТС Маркетолога"
    )


class CatalogChannel(BaseModel):
    """Что рекламодатель может узнать о канале до запуска."""

    channel_id: str
    family: ChannelFamily
    display_name: str = Field(description="абстрактное имя для интерфейса, без названий площадок")
    expected_ecpm_range: tuple[float, float]
    expected_ctr_range: tuple[float, float]
    expected_cvr_range: tuple[float, float]
    daily_unique_capacity_band: tuple[int, int] = Field(
        description="дневной лимит уникальной аудитории сегмента, диапазон"
    )
    supports_video: bool = False
    sms: SmsPublic | None = None
    benchmark_sources: list[str] = Field(default_factory=list)

    @field_validator("expected_ecpm_range", "expected_ctr_range", "expected_cvr_range")
    @classmethod
    def range_is_ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        lo, hi = value
        if lo < 0 or hi < lo:
            raise ValueError("диапазон должен быть неотрицательным и упорядоченным")
        return value

    @property
    def ecpm_mid(self) -> float:
        return (self.expected_ecpm_range[0] + self.expected_ecpm_range[1]) / 2

    @property
    def ctr_mid(self) -> float:
        return (self.expected_ctr_range[0] + self.expected_ctr_range[1]) / 2

    @property
    def cvr_mid(self) -> float:
        return (self.expected_cvr_range[0] + self.expected_cvr_range[1]) / 2

    @property
    def capacity_mid(self) -> float:
        return (self.daily_unique_capacity_band[0] + self.daily_unique_capacity_band[1]) / 2

    @property
    def relative_uncertainty(self) -> float:
        """Полуширина диапазонов относительно середины, усреднённая по трём метрикам.

        Используется планировщиком как ширина коридора вокруг траектории:
        ширина не выдумана, а равна тому, насколько сам каталог не уверен.
        """
        halves = []
        for lo, hi in (self.expected_ecpm_range, self.expected_ctr_range, self.expected_cvr_range):
            mid = (lo + hi) / 2
            halves.append((hi - lo) / (2 * mid) if mid > 0 else 0.0)
        return sum(halves) / len(halves)


class PublicCatalog(BaseModel):
    """Снимок публичных знаний о рынке."""

    catalog_id: str = Field(description="хэш содержимого; seed мира сюда не попадает")
    version: str = "1.0"
    channels: list[CatalogChannel] = Field(min_length=1)

    def by_id(self, channel_id: str) -> CatalogChannel:
        for channel in self.channels:
            if channel.channel_id == channel_id:
                return channel
        raise KeyError(f"канала {channel_id} нет в каталоге")

    @property
    def channel_ids(self) -> list[str]:
        return [c.channel_id for c in self.channels]
