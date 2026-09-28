# Эксперименты v10

Ветка `feature/quality-v10`. Лучший подтверждённый результат платформы: **v9, Recall@50 = 0.900753**. Он сохранён в `experiments/results/v9/answer.csv`. Для v10 результат платформы пока неизвестен.

## Изменения

- Пользователь разрешил использовать `search_category`. Сначала проверен мягкий приоритет; категория 0 не задаёт ограничения. В новом ранкере — совпадение категории и признак объявления услуги.
- Используются все 19 031 обучающих контекста development-протокола; при финальном обучении доступны 26 556 контекстов с положительными объявлениями в корпусе. Оба режима истории сохранены.
- Сохранены широкие кандидаты v9. Добавлены конкуренты, выбранные отдельными моделями с исключением меток текущей группы текстов. Эти модели используют только лексические, метаданные и исходную замороженную E5. Пулы кандидатов исторические, поэтому это не заявляется как полностью вложенный OOF-протокол. Обычный mining v7 также остаётся.
- Замороженная E5 кодирует запрос вместе с текстовыми фильтрами. Добавлены сходство, разница с прежним сходством и ранг. Сам encoder не дообучается в этом эксперименте.
- История объявления: число взаимодействий, разных текстов запросов и взаимодействий из искомой локации. Переходы между локациями учитывают вид услуги.
- Большие обучающие матрицы собираются на диске по частям. Вариант покрытия и варианты с новыми признаками используют одинаковые обучающие примеры.

## Категория

| category_boost | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- |
| 0.0016 | 0.954147 | 0.949338 | 0.950809 |
| 5e-05 | 0.954043 | 0.949044 | 0.950809 |
| 0.0 | 0.954043 | 0.949044 | 0.950809 |
| 0.0001 | 0.954043 | 0.949044 | 0.950809 |
| 0.0002 | 0.954043 | 0.949044 | 0.950809 |
| 0.0004 | 0.954043 | 0.949044 | 0.950809 |
| 0.0008 | 0.954043 | 0.949044 | 0.950809 |

## Ранкер

| variant | weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- | --- |
| quality_deeper | 0.75 | 0.956373 | 0.952132 | 0.952426 |
| quality | 1.0 | 0.956164 | 0.951544 | 0.954338 |
| quality_deeper | 1.0 | 0.956043 | 0.952206 | 0.953088 |
| quality_deeper | 0.5 | 0.955550 | 0.952132 | 0.952132 |
| quality_deeper | 0.3 | 0.955355 | 0.951838 | 0.951985 |
| coverage | 1.0 | 0.955305 | 0.951985 | 0.950956 |
| quality | 0.5 | 0.955052 | 0.951838 | 0.952279 |
| quality | 0.75 | 0.955032 | 0.951838 | 0.953162 |
| coverage | 0.5 | 0.954517 | 0.950662 | 0.952132 |
| coverage | 0.75 | 0.954516 | 0.951250 | 0.951691 |
| quality | 0.15 | 0.954178 | 0.949779 | 0.950809 |
| v9_category | 0.0 | 0.954147 | 0.949338 | 0.950809 |
| coverage | 0.3 | 0.954102 | 0.950074 | 0.950221 |
| quality_deeper | 0.15 | 0.954074 | 0.950074 | 0.950515 |
| quality | 0.3 | 0.953980 | 0.950662 | 0.950809 |
| coverage | 0.15 | 0.953283 | 0.949485 | 0.949926 |

Выбор по development, до просмотра контроля:

```json
{
  "variant": "quality_deeper",
  "weight": 0.75,
  "matched_recall50": 0.9563726551677694,
  "unseen_macro": 0.9521323529411765,
  "held_macro": 0.9524264705882353
}
```

Контроль ранее просматривался, для выбора конфигурации не используется:

```json
{
  "selected_recall50": 0.955,
  "v9_recall50": 0.95,
  "used_for_selection": false
}
```

## Популярность и доступность истории

Отдельная абляция исключает три признака популярности объявления, сохраняя остальные признаки и примеры. Смешанное обучение добавляет группы с удалением 0% и 50% положительных items из разрешённой истории. Исходные группы удаляют 90%; история encoder остаётся строгим подмножеством расширенной истории признаков. Свои тексты/контексты исключены. Положительные метки не удаляются. В смешанной модели доступны четыре сигнала повторных взаимодействий query-item из train и dropout категории с частотой незаданной категории benchmark.

| variant | weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- | --- |
| quality_without_pop | 1.0 | 0.955099 | 0.951691 | 0.953750 |
| quality_without_pop | 0.75 | 0.954554 | 0.951544 | 0.951691 |
| v9_category | 0.0 | 0.954147 | 0.949338 | 0.950809 |
| quality_without_pop | 0.15 | 0.953909 | 0.949485 | 0.950809 |
| quality_without_pop | 0.3 | 0.953889 | 0.950074 | 0.950515 |
| quality_without_pop | 0.5 | 0.953871 | 0.950368 | 0.951250 |

| variant | weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- | --- |
| warm_mixed | 0.75 | 0.954210 | 0.950368 | 0.949338 |
| v9_category | 0.0 | 0.954147 | 0.949338 | 0.950809 |
| warm_mixed | 0.3 | 0.953991 | 0.950074 | 0.951103 |
| warm_mixed | 1.0 | 0.953704 | 0.949191 | 0.950074 |
| warm_mixed | 0.15 | 0.953553 | 0.949779 | 0.950515 |
| warm_mixed | 0.5 | 0.953164 | 0.949191 | 0.949926 |

| variant | weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- | --- |
| warm_deeper | 0.75 | 0.955094 | 0.951250 | 0.950662 |
| warm_deeper | 0.15 | 0.954581 | 0.950368 | 0.950809 |
| warm_deeper | 0.3 | 0.954264 | 0.950662 | 0.950809 |
| v9_category | 0.0 | 0.954147 | 0.949338 | 0.950809 |
| warm_deeper | 1.0 | 0.953538 | 0.950368 | 0.950368 |
| warm_deeper | 0.5 | 0.953211 | 0.950368 | 0.950368 |

| variant | weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- | --- |
| warm_deeper | 0.15 | 0.959934 | 0.954485 | 0.955074 |
| warm_mixed | 0.15 | 0.959747 | 0.954191 | 0.954191 |
| baseline | 0.0 | 0.959726 | 0.953603 | 0.954485 |
| warm_mixed | 0.3 | 0.959685 | 0.953897 | 0.953750 |
| warm_deeper | 0.3 | 0.959635 | 0.954485 | 0.955074 |
| warm_deeper | 0.5 | 0.959522 | 0.955074 | 0.954779 |
| warm_mixed | 0.5 | 0.958301 | 0.953309 | 0.952132 |

Отдельный тёплый ранкер выиграл менее одного попадания на 3400 запросов development, но ухудшил контроль. Поэтому он исключён из первой отправки. Решение принято после просмотра контроля: итоговое значение контроля уже нельзя считать независимой оценкой выбранного состава. Исходный победитель development сохранён в `warm_aux_choice_before_control.json`, исходное падение — в `warm_aux_control.json`.

## Незаданная категория

Категория 0 означает отсутствие ограничения. Дополнительно проверено искусственное обнуление категории у всех validation-запросов; это стресс-проверка, а не новая независимая разметка.

```json
{
  "results": [
    {
      "policy": "v9_without_category_prior",
      "matched_recall50": 0.9540428139856305,
      "unseen_macro": 0.9490441176470589,
      "held_macro": 0.9508088235294118
    },
    {
      "policy": "unspecified_as_mismatch",
      "matched_recall50": 0.9563726551677694,
      "unseen_macro": 0.9521323529411765,
      "held_macro": 0.9524264705882353
    },
    {
      "policy": "unspecified_as_compatible",
      "matched_recall50": 0.9563726551677694,
      "unseen_macro": 0.9521323529411765,
      "held_macro": 0.9524264705882353
    },
    {
      "policy": "actual_queries_wildcard_policy",
      "matched_recall50": 0.9563726551677694,
      "unseen_macro": 0.9521323529411765,
      "held_macro": 0.9524264705882353
    }
  ],
  "benchmark_unspecified_category_fraction": 0.09053833605220228,
  "actual_development_unspecified_queries": 1,
  "weights_fixed": true,
  "source_sha256": "c86e8624e02ad31950aa5a012ee3ce38457d8e5c1877a0034ad4111915312008",
  "policy": "Category 0 means no category constraint; treat compatibility as true.",
  "limitation": "Counterfactual stress proxy, not an independent labeled sample of category-0 searches."
}
```

## Чувствительность к истории

При фиксированных весах и query-encoder заново вычисляются разрешённая история, кандидаты и признаки для долей удаления положительных items 0%, 50%, 90% и 100%. Это проверка доступности истории при inference; она не доказывает устройство скрытого теста. Сравнение относится к основному ранкеру до дополнительных моделей полей/BGE.

| cold_fraction | version | matched_recall50 | unseen_macro | held_macro | pool_unseen | pool_held |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | v9 | 0.953575 | 0.948897 | 0.950809 | 0.995 | 0.995 |
| 0.0 | v10 | 0.953980 | 0.948971 | 0.948971 | 0.995 | 0.995 |
| 0.0 | warm_mixed | 0.954843 | 0.950956 | 0.951103 | 0.995 | 0.995 |
| 0.0 | warm_deeper | 0.956212 | 0.952132 | 0.952721 | 0.995 | 0.995 |
| 0.5 | v9 | 0.954056 | 0.949191 | 0.950809 | 0.995 | 0.995 |
| 0.5 | v10 | 0.954916 | 0.950441 | 0.950662 | 0.995 | 0.995 |
| 0.5 | warm_mixed | 0.954655 | 0.950368 | 0.949926 | 0.995 | 0.995 |
| 0.5 | warm_deeper | 0.955653 | 0.951838 | 0.951250 | 0.995 | 0.995 |
| 0.9 | v9 | 0.954043 | 0.949044 | 0.950809 | 0.995 | 0.995 |
| 0.9 | v10 | 0.956373 | 0.952132 | 0.952426 | 0.995 | 0.995 |
| 0.9 | warm_mixed | 0.954210 | 0.950368 | 0.949338 | 0.995 | 0.995 |
| 0.9 | warm_deeper | 0.955094 | 0.951250 | 0.950662 | 0.995 | 0.995 |
| 1.0 | v9 | 0.953980 | 0.949044 | 0.950809 | 0.995 | 0.995 |
| 1.0 | v10 | 0.957280 | 0.952721 | 0.953309 | 0.995 | 0.995 |
| 1.0 | warm_mixed | 0.954149 | 0.950368 | 0.949338 | 0.995 | 0.995 |
| 1.0 | warm_deeper | 0.955033 | 0.951250 | 0.950662 | 0.995 | 0.995 |

## Представление объявления

| document | query_kind | weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- | --- | --- |
| baseline | plain | 0.0 | 0.954147 | 0.949338 | 0.950809 |
| long_description | plain | 0.025 | 0.953370 | 0.948603 | 0.950515 |
| service_fields | with_filters | 0.025 | 0.953271 | 0.948897 | 0.950221 |
| service_fields | plain | 0.025 | 0.953167 | 0.948603 | 0.950221 |
| long_description | with_filters | 0.025 | 0.952933 | 0.948309 | 0.949926 |
| service_fields | with_filters | 0.05 | 0.952380 | 0.948015 | 0.949926 |
| long_description | with_filters | 0.05 | 0.952348 | 0.947721 | 0.949926 |
| service_fields | plain | 0.05 | 0.952210 | 0.947721 | 0.949926 |
| long_description | plain | 0.05 | 0.952179 | 0.947426 | 0.949926 |
| service_fields | with_filters | 0.1 | 0.952084 | 0.948309 | 0.949926 |
| service_fields | plain | 0.1 | 0.951674 | 0.947426 | 0.949338 |
| long_description | plain | 0.1 | 0.951553 | 0.947426 | 0.949044 |
| long_description | with_filters | 0.1 | 0.951150 | 0.947426 | 0.948750 |

Выбран по development: `{"document": "baseline", "query_kind": "plain", "weight": 0.0, "matched_recall50": 0.9541465603008474, "unseen_macro": 0.9493382352941176, "held_macro": 0.9508088235294118}`.

## Признаки полей объявления в ранкере

| weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- |
| 1.0 | 0.957791 | 0.952721 | 0.954044 |
| 0.75 | 0.957514 | 0.952721 | 0.953750 |
| 0.5 | 0.956969 | 0.953309 | 0.953456 |
| 0.3 | 0.956658 | 0.953015 | 0.953750 |
| 0.15 | 0.956398 | 0.952132 | 0.953309 |
| 0.0 | 0.956373 | 0.952132 | 0.952426 |

Выбран по development: `{"weight": 1.0, "matched_recall50": 0.9577905827668249, "unseen_macro": 0.9527205882352942, "held_macro": 0.9540441176470589}`.

Контроль: `{"baseline_recall50": 0.955, "selected_recall50": 0.955, "used_for_selection": false}`.

## Дообучение cross-encoder

| weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- |
| 0.1 | 0.955052 | 0.950956 | 0.952279 |
| 0.05 | 0.954660 | 0.950368 | 0.951103 |
| 0.0 | 0.954147 | 0.949338 | 0.950809 |
| 0.025 | 0.954013 | 0.949191 | 0.950809 |

Выбран по development: `{"weight": 0.1, "matched_recall50": 0.9550520205052287, "unseen_macro": 0.9509558823529412, "held_macro": 0.9522794117647059}`.

Контроль: `{"baseline": 0.95, "selected": 0.9566666666666667, "used_for_selection": false}`.

## Независимый encoder BGE-M3

| kind | variant | weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- | --- | --- |
| ranker | deep | 1.0 | 0.959726 | 0.953603 | 0.954485 |
| ranker | deep | 0.3 | 0.959590 | 0.955368 | 0.955221 |
| ranker | deep | 0.75 | 0.959502 | 0.954191 | 0.955368 |
| ranker | deep | 0.5 | 0.958846 | 0.954191 | 0.954632 |
| ranker | compact | 0.75 | 0.958010 | 0.953309 | 0.954926 |
| ranker | deep | 0.15 | 0.957972 | 0.953309 | 0.954338 |
| ranker | compact | 0.5 | 0.957861 | 0.953603 | 0.954338 |
| baseline | none | 0.0 | 0.957791 | 0.952721 | 0.954044 |
| ranker | compact | 0.3 | 0.957727 | 0.952426 | 0.954632 |
| direct | none | 0.025 | 0.957556 | 0.952721 | 0.954044 |
| ranker | compact | 1.0 | 0.957546 | 0.952426 | 0.953750 |
| ranker | compact | 0.15 | 0.957524 | 0.952426 | 0.954044 |
| direct | none | 0.05 | 0.956269 | 0.951838 | 0.953162 |
| direct | none | 0.1 | 0.955754 | 0.951544 | 0.953603 |

Выбран по development: `{"kind": "ranker", "variant": "deep", "weight": 1.0, "matched_recall50": 0.9597256528572226, "unseen_macro": 0.9536029411764706, "held_macro": 0.954485294117647}`.

Контроль: `{"baseline_recall50": 0.955, "selected_recall50": 0.9583333333333334, "used_for_selection": false}`.

## Отправка и воспроизведение

Ответ: `experiments/results/v10/answer.csv`. SHA-256: `f75d1a1358a1a1c2b40d30d608c4f8a56e88fada5b0d186b203f023b301fb0be`. Notebook: `experiments/Avito_v10_candidate.ipynb`; архив с локальными весами: `deliverables/avito_v10_solution.zip`.

Свежий запуск cpu без сети и готового CSV: 380.09 с, 30 ячеек, побайтовое совпадение ответа.

## Повторение экспериментов

Все команды выполняются из корня проекта. Основной notebook воспроизводит inference; следующие команды продолжают обучение из исследовательского workspace v9 и явно запускают эксперименты на локальных данных. Для обучения с нуля сначала нужны базовые компоненты v5/v7 и шесть OOF query-encoder, описанные в предыдущих отчётах и в `query_encoder_rank.py`; `context_scale_v9.py` восстанавливает обучающие пулы v9. Эти крупные обучающие кэши не включены в архив inference. Большие промежуточные матрицы не требуются для обычного Run All.

```text
python .development/quality_v10.py
python .development/train_quality_v10.py
python .development/no_popularity_v10.py
python .development/warm_history_v10.py
python .development/warm_deeper_v10.py
python .development/check_category_shift_v10.py
python .development/semantic_fields_v10.py
python .development/field_ranker_v10.py
python .development/history_sensitivity_v10.py
python .development/prepare_bge_m3_v10.py
python .development/encode_bge_m3_v10.py
python .development/bge_ranker_v10.py
python .development/finetune_cross_encoder_v10.py
python .development/export_quality_v10.py
python .development/validate_quality_v10.py
python .development/package_quality_v10.py
python .development/verify_quality_v10.py
```

BGE-M3 — `BAAI/bge-m3`, MIT, фиксированная ревизия `5617a9f61b028005a4858fdac845db406aefb181`. Официальное описание: https://huggingface.co/BAAI/bge-m3. Нормированный CLS-вектор, без префиксов E5. Пилот: `prepare_bge_m3_v10.py` → `encode_bge_m3_v10.py` → `bge_ranker_v10.py`. Загрузка скачивает только публичные веса; кодирование и обучение выполняются локально. Запросы и объявления не передаются внешнему сервису. Замороженные query-векторы вычислены из текста и фильтров, без query_id или test-разметки. Если канал выбран, его веса и векторы включены в архив; notebook использует сохранённые векторы для точного CPU-воспроизведения.

## Cross-encoder в итоговой комбинации

| weight | matched_recall50 | unseen_macro | held_macro |
| --- | --- | --- | --- |
| 0.0 | 0.959934 | 0.954485 | 0.955074 |
| 0.2 | 0.959309 | 0.954118 | 0.954265 |
| 0.025 | 0.959301 | 0.953897 | 0.954779 |
| 0.1 | 0.958924 | 0.953603 | 0.955074 |
| 0.05 | 0.958803 | 0.953603 | 0.954926 |

Этот пилот сравнивался с вариантом до контрольного исключения дополнительного ранкера истории; пилот не выбран, потому что ухудшил development при всех ненулевых весах.

Измеряется дополнение к уже выбранному v10, а не только к v9. Development-счёты вычислены локально CUDA BF16. При включении канала notebook действительно обрабатывает query и документы моделью на top-200 v9+category; готовые test-ответы или test-логиты не подставляются. CUDA BF16 повторяет режим пилота. CPU float32 поддерживается, но существенно медленнее; побайтовая проверка указывает фактически проверенное устройство.

## Оставшиеся ошибки

Пропуски известных положительных разделены на непопадание в широкий пул и ошибки выбора top-50. Источник меток — только train/development. Подробности: `artifacts/quality-v10/development_misses.csv`. Не выбранные пользователем объявления не объявляются доказанными отрицательными.

```json
{
  "counts": [
    {
      "mode": "unseen_text",
      "queries": 3400,
      "queries_with_missed_positive": 157,
      "retrieval_missed_positive_items": 17,
      "ranking_missed_positive_items": 140
    },
    {
      "mode": "held_context",
      "queries": 3400,
      "queries_with_missed_positive": 156,
      "retrieval_missed_positive_items": 17,
      "ranking_missed_positive_items": 139
    }
  ],
  "gold_source": "development derived from train only; no benchmark labels",
  "unselected_items_not_proven_irrelevant": true,
  "cross_validation_precision": "Cached CUDA BF16 pilot if selected",
  "primary_model_top_features_by_gain": [
    {
      "feature": "log_baseline_rank",
      "gain_share": 0.6589892317642322
    },
    {
      "feature": "baseline_score",
      "gain_share": 0.06689666101931634
    },
    {
      "feature": "learned_log_geo_rank",
      "gain_share": 0.04811209852663203
    },
    {
      "feature": "log_distance_km",
      "gain_share": 0.02646904051564933
    },
    {
      "feature": "log_e5_geo_rank",
      "gain_share": 0.02374063778061832
    },
    {
      "feature": "body_token_coverage",
      "gain_share": 0.015174039059177595
    },
    {
      "feature": "geo_compatibility",
      "gain_share": 0.013560693017242274
    },
    {
      "feature": "log_reviews",
      "gain_share": 0.00999465845946462
    },
    {
      "feature": "context_log_distance",
      "gain_share": 0.009735204024589135
    },
    {
      "feature": "log_body_rank",
      "gain_share": 0.008691703067866072
    },
    {
      "feature": "neighbor_centroid_cosine",
      "gain_share": 0.0077897845962459295
    },
    {
      "feature": "body_idf_coverage",
      "gain_share": 0.006890355718876333
    }
  ],
  "importance_is_descriptive_not_causal": true,
  "ranking_misses_same_location_fraction": 0.4014336917562724
}
```

## Ограничения

Development и контроль использовались в прежних экспериментах, они не являются новым независимым тестом. Сохранено известное ограничение старых вспомогательных priors v4. Удаление 90% положительных items из истории — допущение, а не известное устройство benchmark. Невыбранные объявления не считаются доказанно нерелевантными. Метрику скрытого теста может установить только платформа.

Сторонний код или ответы кандидатов не копируются. Используются локальные open-source E5, mMARCO cross-encoder, Transformers, PyTorch, LightGBM и библиотеки из requirements.txt.
