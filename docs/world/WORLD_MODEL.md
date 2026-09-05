# Модель мира и симулятор рынка

Статус: архитектурный контракт для совместной разработки.  
Дата: 03.09.2026.  
Обновление 05.09.2026: API 1.1 и реализованные расширения описаны в
[`EXTENSIONS.md`](EXTENSIONS.md); этот документ уточняет нижеуказанные OPEN
для аудитории, фрода, недельной квоты SMS и сценарных компаний.
Продуктовая постановка: [`Project description.md`](../Project%20description.md).
Краткая визуальная версия: [`world_model_market_simulator.pdf`](../output/pdf/world_model_market_simulator.pdf).

## 1. Назначение

Модель мира создаёт синтетический, изменчивый и воспроизводимый рекламный рынок.
Симулятор позволяет сравнивать стратегии почасового управления медиапланом без
подключения реальных рекламных кабинетов.

Главный критерий реализма: не визуальное сходство с интерфейсом площадки, а
корректная реакция рынка на действие оптимизатора:

- увеличение лимита может дать больше показов, но эффект насыщается;
- доступный инвентарь и аудитория конечны;
- цена, спрос и отклик меняются по времени;
- одинаковые условия дают воспроизводимый эксперимент;
- будущие параметры и шоки не видны оптимизатору.

Модель не претендует на прогноз конкретной рекламной сети. Это испытательный
стенд для алгоритмов планирования и управления.

## 2. Термины и компоненты

### World model

Генеративная модель истинного рынка. Хранит скрытые параметры каналов, сегменты,
ёмкости, временные профили, latent regime, накопленное насыщение и сценарий.

### Simulator engine

Исполняет один час: валидирует action, получает экзогенную случайность,
применяет ограничения, рассчитывает outcomes и обновляет состояние.

### Public catalog

Доступные планировщику ожидания: диапазоны CPM/CTR/CR, ориентировочная ёмкость,
поддерживаемые KPI и допустимые лимиты. Каталог не равен истинным параметрам
конкретного эпизода.

### Observation

Почасовой агрегированный факт, который могла бы вернуть рекламная система.

### Execution service / optimizer

Внешний по отношению к миру компонент. Хранит утверждённый план, сравнивает его с
фактом и задаёт лимиты следующего часа.

## 3. Информационная граница

```text
PUBLIC                         PRIVATE
------                         -------
catalog expectation bands     true channel parameters
approved media plan           latent market regime
past observations             future shocks
remaining campaign budget     future exogenous noise
current hour                  exact remaining inventory/audience

Optimizer -> Action -> Simulator -> Observation -> Optimizer
```

Код оптимизатора не должен импортировать `WorldState`, `ShockState`, внутренние
конфиги распределений или debug snapshot. Это важнее удобства локальной
разработки: иначе стратегия случайно обучится на данных, которых не будет в
реальном контуре.

## 4. Горизонт и каналы

- Шаг: 1 час.
- Горизонт: 14–21 день, то есть 336–504 шага.
- Каналов: 8.
- Гранулярность: один агрегированный пакет на канал в час.
- Целевая производительность: полный эпизод на ноутбуке за секунды.

Вместо восьми независимых реализаций используются три профиля:

| Профиль | Каналы | Основное поведение |
|---|---|---|
| `auction` | 3 social + programmatic | плавающий eCPM, широкий охват, плавное насыщение |
| `marketplace` | 3 marketplace | высокий intent, ограниченный инвентарь, более сильная зависимость CVR |
| `direct` | SMS | цена контакта близка к фиксированной, жёсткая база, доставляемость и fatigue |

Различия между каналами задаются конфигурацией параметров и кривых, а не
копированием бизнес-логики.

## 5. Скрытое состояние мира

Минимальное состояние часа `t`:

```python
WorldState = {
    "hour": int,
    "channel_state": {
        channel_id: {
            "latent_demand": float,
            "latent_ecpm": float,
            "latent_ctr": float,
            "latent_cvr": float,
            "inventory_remaining": float,
            "unique_capacity_remaining": float,
            "saturation": float,
            "fatigue": float,
            "regime": str,
        }
    },
}
```

Это логическая схема, а не обязательное имя Python-класса. Реализация может
хранить массивы NumPy или dataclasses, если публичный контракт остаётся тем же.

## 6. Как разыгрывается один час

Базовая последовательность:

1. Проверить action и остаток общего бюджета.
2. Получить случайные значения по стабильным ключам часа и канала.
3. Применить активные shocks к скрытым параметрам.
4. Сгенерировать доступный трафик и инвентарь.
5. Рассчитать эффективную цену и допустимый объём закупки.
6. Рассчитать показы и уникальный охват.
7. Разыграть клики и конверсии.
8. Обновить saturation, fatigue, остатки и режим.
9. Сформировать безопасный observation.

Минимальная проверяемая механика может начинаться со следующих моделей:

```text
requests_t ~ Poisson(base_volume × hourly_profile × seasonality × shock)

impressions_t <= min(requests_t, inventory_t, affordable_volume(cap_t, eCPM_t))

clicks_t ~ Binomial(impressions_t, clipped_CTR_t)

conversions_t ~ Binomial(clicks_t, clipped_CVR_t)
```

`effective_CTR`, `effective_CVR` и/или `effective_eCPM` зависят от часа,
насыщения, fatigue и скрытого режима. Точную форму кривых нужно хранить в
конфигурации и проверять отдельно.

Для MVP допустимы простые монотонные Hill/logistic-кривые. Нейросеть и обучение
latent dynamics не нужны, пока нет данных, на которых их можно валидировать.

## 7. Инварианты

Для каждого канала и часа:

```text
0 <= spend <= spend_cap
0 <= impressions <= requests
0 <= unique_reach <= impressions
0 <= clicks <= impressions
0 <= conversions <= clicks
eCPM >= 0
```

Для кампании:

```text
cumulative_spend <= total_budget
current_hour <= horizon_hours
```

Не исправляйте нарушение инварианта молча через `abs`, неявный `clip` или
пересчёт результата. Ошибка должна быть локализована в модели, которая создала
некорректное значение. Ограничение вероятностей в `[0, 1]` является частью
явного контракта response model.

## 8. Публичный контракт

### Reset

```python
observation, info = simulator.reset(
    seed_bundle=SeedBundle(
        catalog_seed=...,
        world_seed=...,
        noise_seed=...,
    ),
    scenario_id="stable",
)
```

Упрощённый `reset(seed)` допустим, если он детерминированно создаёт именованные
sub-seeds и сохраняет их в метаданных эксперимента.

`reset` обязан:

- создать чистое состояние эпизода;
- зафиксировать версии каталога, мира и сценария;
- вернуть начальное наблюдение без будущей информации;
- не зависеть от ранее выполненных эпизодов.

### Action

```json
{
  "spend_caps": {
    "social_1": 10000.0,
    "social_2": 8000.0,
    "social_3": 7000.0,
    "programmatic": 15000.0,
    "marketplace_1": 12000.0,
    "marketplace_2": 9000.0,
    "marketplace_3": 6000.0,
    "sms": 5000.0
  }
}
```

Правила валидации MVP:

- action содержит точный набор активных каналов;
- значения конечны, неотрицательны и выражены в рублях;
- сумма caps не превышает доступный остаток кампании;
- неизвестный канал или невалидный action вызывает явную ошибку;
- симулятор не нормализует и не перераспределяет action молча.

Если позднее появятся ставки, bid multipliers или минимальные закупочные пакеты,
они добавляются версионированным расширением schema, а не переопределением
смысла `spend_caps`.

### Step

```python
observation, metrics, terminated, info = simulator.step(action)
```

Один вызов завершает текущий час и переводит время на следующий. Повторный
`step` после `terminated=True` является ошибкой до нового `reset`.

### Observation

```json
{
  "hour": 12,
  "by_channel": {
    "social_1": {
      "requests": 120000,
      "impressions": 84000,
      "unique_reach": 61000,
      "clicks": 1050,
      "conversions": 73,
      "spend": 9870.50,
      "ecpm": 117.51
    }
  }
}
```

`metrics` может дополнительно содержать безопасные накопительные итоги. Плановая
траектория не добавляется симулятором: сервис исполнения объединяет observation
с медиапланом самостоятельно.

### Info

Допустимые поля public `info`:

- версия API;
- `episode_id`, `scenario_id`, hashes конфигураций;
- список применённых публичных ограничений;
- сообщения валидации;
- причина завершения.

Недопустимые поля: истинные CTR/CVR/eCPM, latent regime, будущие shocks и
предсказанный oracle outcome.

### Reward и Gymnasium

Доменный симулятор возвращает факты и не зашивает одну целевую функцию. Отдельный
адаптер может представить его как Gymnasium environment:

```text
observation, reward, terminated, truncated, info = gym_env.step(action)
```

Reward вычисляется конфигурацией эксперимента: conversions, revenue, штраф за
CPA/ROAS, отклонение от траектории или их комбинация. Благодаря этому одна
физика мира обслуживает задачи планирования типа A, типа B и исполнения.

## 9. Seeds и common random numbers

Одного глобального последовательного RNG недостаточно. Стратегии принимают
разные действия, поэтому ветвящаяся логика может потребить разное число
случайных значений и разрушить paired comparison.

Случайные значения должны адресоваться стабильным ключом:

```text
(noise_seed, hour, channel_id, event_type)
```

Примеры `event_type`: `traffic`, `price`, `reach`, `click`, `conversion`.
Практически это можно реализовать через заранее созданную exogenous tape или
через независимые детерминированные sub-generators.

Для сравнения двух стратегий фиксируются:

- версия public catalog;
- `world_seed`;
- `noise_seed`;
- `scenario_id` и версия сценария;
- горизонт и общий бюджет.

Меняется только policy. Результат сохраняет полный seed bundle и hashes
конфигураций, чтобы прогон можно было повторить.

## 10. Шоковые сценарии

Сценарий декларативен и не содержит исполняемого кода:

```yaml
id: marketplace_cpm_spike
events:
  - start_hour: 168
    duration_hours: 48
    target_channels: [marketplace_1]
    parameter: ecpm
    multiplier: 1.40
    recovery: linear
```

Минимальный набор:

- stable/no-shock;
- резкий рост CPM;
- падение CTR;
- падение CVR;
- временное исчерпание инвентаря;
- полная приостановка канала;
- всплеск спроса;
- постепенное recovery.

Оптимизатор не получает расписание событий. Тестовый runner знает scenario для
организации эксперимента, но policy видит только последствия в observations.

## 11. Публичный каталог и калибровка

Каталог содержит ожидания, а не oracle truth:

```json
{
  "channel_id": "social_1",
  "family": "auction",
  "expected_ecpm_range": [90, 150],
  "expected_ctr_range": [0.008, 0.018],
  "expected_cvr_range": [0.025, 0.055],
  "daily_unique_capacity_band": [200000, 350000]
}
```

Истинные параметры эпизода выбираются внутри этих диапазонов или рядом с ними,
но не обязаны совпадать с серединой. Это создаёт controllable uncertainty без
обмана о точности реальной площадки.

Этапы калибровки:

1. Экспертные синтетические диапазоны и sanity checks.
2. Проверка распределений на публичных агрегированных benchmark-значениях.
3. Если появятся собственные агрегированные логи — оценка response curves и
   posterior predictive checks.

На этапе хакатона маркетинговое mix-моделирование не входит в scope. Из Meridian
и Robyn берётся только идея насыщаемых response curves и будущего процесса
калибровки.

## 12. Контракт сервиса исполнения

Симулятор и оптимизатор соединяет execution loop:

```python
observation, info = simulator.reset(seed_bundle, scenario_id)

while not done:
    optimizer_state = execution_service.combine(
        plan=approved_plan,
        observation=observation,
        history=history,
        remaining_budget=remaining_budget,
    )
    action = optimizer.propose_action(optimizer_state)
    observation, metrics, done, info = simulator.step(action)
    history.append(observation)
```

Это позволяет заменить optimizer без изменения мира: static rule, proportional
pacing, PID, heuristic, bandit или RL policy используют один интерфейс.

## 13. Baselines и оценка

Обязательные стратегии:

1. `static` — исходные доли бюджета не меняются;
2. `proportional_pacing` — расход по оставшемуся бюджету и времени;
3. `pid` — коррекция по ошибке относительно накопительной траектории;
4. `greedy_marginal` — перераспределение по наблюдаемой предельной отдаче;
5. `oracle` — только в тестовом runner как недостижимая верхняя граница.

Основная оценка выполняется минимум на 30 paired seeds:

- конечное отклонение основного KPI от плана;
- MAPE накопительного spend/KPI согласно постановке;
- WAPE или абсолютное отклонение для рядов около нуля;
- итоговые clicks/conversions и использование бюджета;
- нарушения constraints;
- paired delta относительно static baseline;
- время обнаружения шока и время восстановления;
- среднее, стандартное отклонение и доверительный интервал;
- худший дециль или CVaR для оценки устойчивости.

Один красивый seed не считается доказательством улучшения.

## 14. Тестовый контракт

### Детерминизм

- два одинаковых reset + последовательности actions дают побитово или численно
  одинаковые observations;
- reset не зависит от истории предыдущих эпизодов;
- сериализованный experiment manifest воспроизводит прогон.

### Физические свойства

- нулевой cap даёт нулевой spend;
- увеличение cap не уменьшает ожидаемый объём закупки до насыщения;
- очень большой cap упирается в inventory/capacity;
- saturation снижает предельную отдачу;
- SMS не превышает остаток базы;
- все инварианты выполняются на каждом шаге.

### Сценарии

- shock включается на точной границе;
- применяется только к целевым каналам;
- завершается и восстанавливается согласно recovery;
- информация о будущем shock отсутствует в public outputs.

### Интеграция

- неизвестный канал, NaN, отрицательный cap и overspend action отклоняются;
- все baseline policies проходят полный эпизод;
- прогон 21 × 24 × 8 укладывается в заданный runtime;
- JSON/CSV export содержит полный часовой журнал.

## 15. Происхождение решений

| Что используем | Источник | Как адаптируем |
|---|---|---|
| 8 каналов, три профиля, shocks, инварианты, почасовой шаг | Внутренняя постановка: `Project description.md` и презентация «Симулятор рынка — основные идеи» | Считаем исходными требованиями команды |
| `reset/step`, state-action-observation, конфигурируемые модели цены/CTR/CVR | [RTBGym / SCOPE-RL](https://github.com/hakuhodo-technologies/scope-rl/tree/main/rtbgym) | Берём форму интерфейса; вместо bid-adjustment action используем вектор часовых spend caps |
| Pacing, hard throttling и Mystique baseline | [Yahoo Budget Pacing Simulation](https://github.com/yahoo/BudgetPacingSimulation) | Используем как ориентир baseline-контроллеров, не копируем весь движок |
| ALM, TA-PID, M-PID, Mystique, BROI и единый benchmark | [BAT](https://github.com/avito-tech/bat-autobidding-benchmark) | Берём набор сравнительных политик и структуру оценки |
| Воспроизводимая offline-оценка и конфигурируемые эксперименты | [AuctionGym](https://github.com/amazon-science/auction-gym) | Берём seeded configs, repeated runs и журнал метрик; микро-аукционы не переносим в MVP |
| Явные сущности, probabilistic behavior и uncertainty | [RecSim NG](https://github.com/google-research/recsim_ng) | Берём разделение state/transition/observation; TensorFlow/Edward2 не требуются |
| Многоканальность, saturation, non-stationarity и change-point framing | [Adaptive Budget Optimization for Multichannel Advertising](https://arxiv.org/abs/2502.02920) | Используем для дизайна response curves, шоков и метрик adaptation/regret |
| Иерархия «распределение общего бюджета -> канальное исполнение» и channel capacity | [HiBid](https://arxiv.org/abs/2312.17503) | Используем архитектурную границу; offline deep RL не является требованием MVP |
| Saturation/adstock и сценарная аллокация бюджета | [Google Meridian](https://github.com/google/meridian), [Meta Robyn](https://github.com/facebookexperimental/Robyn) | Используем как будущий путь калибровки агрегированных response curves; полноценный MMM вне scope |

Проекты имеют разные лицензии. До прямого переноса кода нужно проверить LICENSE,
атрибуцию и NOTICE конкретного репозитория. Архитектурные идеи и ссылки можно
использовать независимо от решения о кодовом заимствовании.

## 16. Что сознательно не переносим

- Из AuctionGym/BAT/AuctionNet: микро-аукцион каждого impression.
- Из RecSim NG: тяжёлый probabilistic programming runtime.
- Из HiBid: иерархическую offline DRL-инфраструктуру.
- Из Meridian/Robyn: полноценную MMM и причинную атрибуцию.
- Из реальных рекламных платформ: интерфейсы, названия и обещание platform-level
  accuracy.

Это не недостатки будущей системы, а границы проверяемого MVP.

## 17. Открытые решения команды

Агенты не должны закрывать эти пункты скрытыми предположениями. Решение нужно
зафиксировать в этом документе и тестах.

- `OPEN`: окончательные `channel_id` и единицы денежных значений API.
- `OPEN`: конкретные диапазоны CPM/CTR/CR и источник каждого диапазона.
- `OPEN`: формула unique reach и способ учёта повторных показов.
- `OPEN`: входит ли VTR в обязательный MVP schema или только в video-каналы.
- `RESOLVED 2026-09-05`: аудитории имеют попарные общие пулы и симметричную
  настраиваемую матрицу; независимость включается через default_overlap=0.
  Тройные пересечения не моделируются. Подробности: `EXTENSIONS.md`.
- `OPEN`: нужны ли минимальные пакеты и cooldown для SMS.
- `OPEN`: будут ли conversions мгновенными или появится observation delay.
- `OPEN`: стратегия получает полный почасовой requests или только delivery
  metrics, доступные условному рекламному кабинету.
- `OPEN`: точная функция reward для демонстраций типов A, B и исполнения.

## 18. Definition of Done модели мира

Модель мира готова к интеграции, когда:

- публичные schemas версионированы и задокументированы;
- 8 каналов работают через 3 профиля;
- реализованы stable и минимум четыре shock-сценария;
- выполняются инварианты и no-leakage tests;
- paired runs действительно используют одну exogenous tape;
- static, pacing и PID подключаются без импорта внутренних классов мира;
- полный 21-дневный прогон выполняется за секунды;
- experiment manifest полностью воспроизводит результат;
- отчёт показывает распределения метрик минимум по 30 seeds, а не только один
  демонстрационный эпизод.
