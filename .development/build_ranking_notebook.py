"""Assemble the complete ranking solution into ordinary notebook cells.

Numerical functions are copied from the controlled experiment; development
helpers are not imported by the resulting notebook.
"""
import ast,json,re,textwrap
from pathlib import Path
root=Path(__file__).resolve().parents[1]
doc=json.loads((root/'Avito_v0.3.ipynb').read_text(encoding='utf-8'))
codes=[''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code']
experiment=(root/'.development/ranking_experiment.py').read_text(encoding='utf-8')
selection=(root/'.development/ranking_selection.py').read_text(encoding='utf-8')
parts=[]
def md(text):parts.append('# %% [markdown]\n'+'\n'.join('# '+line if line else '#' for line in text.strip().splitlines())+'\n')
def code(text):parts.append('# %%\n'+text.strip()+'\n')
def defs(text,names=None):
    tree=ast.parse(text)
    return '\n\n'.join(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names))

md('''# Кандидатогенерация услуг Авито: ranking v0.4.0

BM25, символьный TF-IDF и локальная E5 создают пул; обучаемая модель отбирает 50 объявлений.
Сравниваются ranking-loss и бинарная классификация, 4000 и расширенная выборка контекстов.
В обучение добавлены трудные отрицательные примеры прежней модели. Выбор учитывает
знакомые/новые тексты, длину запроса, фильтры и наличие локации в корпусе.

Все вычисления локальны. Код решения целиком находится здесь; внешние API и разметка
benchmark не используются. `search_category` не входит в scoring.
Запуск: три Parquet в этой папке, зависимости из requirements.txt, локальные модели,
**Restart Kernel → Run All**. Результат — answer.csv; артефакты — artifacts/ranking-v1.
Прежние небольшие бустинги поставляются как замороженные priors. Их полная обучающая
рецептура сохранена в Avito_v0.3.ipynb; новые модели обучаются в этом ноутбуке.''')
first=codes[0].replace('"version": "semantic-v1"','"version": "ranking-v1"')
first=first.replace('import gc','if (ROOT / ".ranking_deps").is_dir():\n    sys.path.insert(0, str(ROOT / ".ranking_deps"))\n\nimport ast\nimport gc',1)
first+='\nfrom catboost import CatBoostRanker, CatBoostClassifier\nimport lightgbm as lgb\n'
first+='\n# The search was completed on development before opening the 800-query audit.\n# Default Run All trains the frozen winner; opt in to repeat the full comparison.\nRUN_MODEL_SEARCH = os.getenv("AVITO_RUN_MODEL_SEARCH", "0") == "1"\n'
code(first)
md('''## 1. Данные и идентификаторы

ID сохраняются строками, корпус сортируется по item_id. Тексты пропусков пустые,
Decimal-цены/координаты переводятся в числовые массивы. Порядок документов фиксирован.''')
code(codes[1])
md('''## 2. Нормализация и состав данных

Одинаковая нормализация применяется к train и benchmark. Полный контекст включает
текст, локацию, фильтры, доставку и категорию для сопоставления labels; категория
не становится входом обучаемой модели. Повторные пары контекст/item_id удаляются.''')
code(codes[2])
md('''## 3. Исторические выборки: теперь development

Прежние 1400 validation и два среза по 600 запросов уже анализировались при разработке.
Все 2600 теперь используются для выбора новой модели. Они исключены из истории
нового обучения. Дополнительные 800 текстов будут зарезервированы до любого нового fit.
Метки ограничены положительными объявлениями, присутствующими в доступном корпусе.''')
code(codes[3])
md('''## 4. Лексические индексы

BM25 заголовка и полного текста, русский Snowball stemmer, символьный TF-IDF.
Индексы строятся без labels по всем объявлениям корпуса. Кеш необязателен;
AVITO_REBUILD_CACHE=1 перестраивает производные данные и новые модели.''')
code(codes[4])
md('''## 5. География и перенос намерения из истории

Точное совпадение города, расстояние и обученные переходы локаций — мягкие признаки.
Похожие тексты истории задают распределение подкатегорий. Целевые взаимодействия
оценочных запросов удаляются из истории до построения их признаков.''')
history_source=codes[5].replace('CACHE / "fresh_audit_protocol.json"','CACHE / "prior_split_protocol.json"')
history_source=history_source.replace('(CACHE / "prior_split_protocol.json").write_text',
    'fresh_protocol["role"] = "historical_v0.3_slice_now_development"\nfresh_protocol["used_for_model_selection"] = True\n(CACHE / "prior_split_protocol.json").write_text')
code(history_source)
md('''## 6. Исходные кандидаты и 38 признаков

Лексические источники сохраняются; позднее добавляются E5 cosine и cosine с географией.
Обучаемый отбор использует текстовые оценки/ранги/покрытие, географию, подкатегории,
рейтинг, отзывы, цену и флаги связи. Сырые ID и search_category в модели отсутствуют.''')
code(defs(codes[6])+'\nbest_config = '+repr({'name':'hybrid_19','title':.3,'body':.5,'char':.2,'filter':.03,'geo':.5,'micro':.5}))
code(codes[8].split('def build_oof_training')[0]+'\n'+defs(codes[9],{'build_rank_feature_records','model_scores','blend_predictions'}))
md('''## 7. Локальный семантический энкодер

intfloat/multilingual-e5-small, MIT, фиксированная ревизия и контрольные суммы.
query:/passage:, attention-mask mean pooling, L2-нормировка, максимум 192 токена.
Энкодер не дообучается; эмбеддинги не используют поведенческие labels. Точный поиск
по корпусу выполняется на CPU с float64-накоплением. GPU нужен для кодирования
при отсутствии приложенных векторов.''')
code(codes[12].split('semantic_index = SemanticIndex()')[0])
aliases='''legacy_retrieve_features = retrieve_features
legacy_rank_features = rank_features
legacy_score_candidates = score_candidates
legacy_negative_sample = hard_negative_sample
LEGACY_FEATURE_NAMES = list(RANK_FEATURE_NAMES)
SEMANTIC_FEATURE_NAMES = ["e5_cosine", "e5_geo_affinity", "from_lexical_pool", "from_semantic_pool", "log_e5_rank", "log_e5_geo_rank"]
RANK_FEATURE_NAMES = LEGACY_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES
'''
code(aliases+defs(codes[13])+'''
retrieve_features = retrieve_semantic_features
rank_features = rank_semantic_features
score_candidates = score_lexical_columns
hard_negative_sample = semantic_negative_sample
''')
md('''## 8. Замороженные priors v0.3.0

Четыре небольших HGB-модели обучены на предоставленном train, не на разметке benchmark.
Оценочные модели на 4000 текстах используются для reference и поиска трудных negatives.
Две финальные модели на 6600 текстах используются только при итоговом benchmark inference.
Вес новой семантической модели в прежнем ансамбле — 25%, прежнего отбора — 75%.
Файлы и их происхождение проверяются по SHA-256. Рецептура — Avito_v0.3.ipynb.''')
code('''PRIORS_DIR = ROOT / "models" / "retrieval-priors"
prior_manifest = json.loads((PRIORS_DIR / "manifest.json").read_text(encoding="utf-8"))
assert prior_manifest["input_sha256"] == input_hashes
for name, expected in prior_manifest["sha256"].items():
    assert sha256_file(PRIORS_DIR / name) == expected, f"Modified frozen prior: {name}"
prior_legacy = joblib.load(PRIORS_DIR / "evaluation_legacy.joblib")
prior_semantic = joblib.load(PRIORS_DIR / "evaluation_semantic.joblib")
''')
md('''## 9. Новое разбиение и расширение контекстов

Новые 800 текстов не входят в прежние прямые обучающие запросы; их положительные ID
также не были прямыми positives оценочных priors. Тексты и 90% целевых ID удаляются
из истории до нового обучения. До четырёх дополнительных контекстов текста расширяют
географию/фильтры; прежний прямой обучающий контекст сохраняется отдельно.
Фактический объём определяется доступными положительными объявлениями корпуса;
20000 — верхняя граница, а не обещание наличия 20000 размеченных примеров.''')
prefix=experiment[experiment.index('EXPERIMENT_CONFIG ='):experiment.index('dataset=grouped_training')]
start=prefix.index('fingerprint = hashlib.sha256')
end=prefix.index('def context_sample')
key='''# Revision covers the copied numerical retrieval/feature/sampling routines.
# Changing those routines requires incrementing data_revision; full rebuild ignores caches.
fingerprint = hashlib.sha256(json.dumps({"inputs": input_hashes,
    "config": EXPERIMENT_CONFIG, "baseline_config": CONFIG, "semantic": SEMANTIC_CONFIG,
    "model": model_manifest, "priors": prior_manifest["sha256"], "data_revision": 1},
    sort_keys=True).encode()).hexdigest()[:16]

'''
prefix=prefix[:start]+key+prefix[end:]
old=prefix[prefix.index('source_item_path ='):prefix.index('def prior_scores')]
new='''semantic_index = SemanticIndex()
all_encoder_queries = pd.concat([expanded_training.query_norm, development.query_norm,
    new_audit.query_norm, queries.query_norm], ignore_index=True)
semantic_index.prepare(all_encoder_queries)
embedding_checksums = {"items": sha256_file(semantic_index.item_path),
    "queries": sha256_file(semantic_index.query_path)}
(CACHE / "embedding_manifest.json").write_text(json.dumps({"model": model_manifest,
    "config": SEMANTIC_CONFIG, "sha256": embedding_checksums}, indent=2), encoding="utf-8")

'''
prefix=prefix.replace(old,new).replace('if path.exists():return joblib.load(path)','if USE_CACHE and path.exists():return joblib.load(path)')
# Separate initialization from the OOF builder so the narrative is easy to follow.
split=prefix.index('def grouped_training')
code(prefix[:split])
md('''## 10. OOF-признаки и трудные отрицательные примеры

Все контексты одного нормализованного текста относятся к одной OOF-группе.
При построении группы её тексты и 90% positives исключаются из вспомогательной истории.
Negatives: лексические/семантические лидеры, высокие оценки замороженного бустинга,
случайный хвост. Пропущенные поиском positives не добавляются в пул принудительно.
Ранги рассчитываются по полному пулу до sampling. Ненаблюдённые взаимодействия —
слабые отрицательные примеры; они не являются гарантированно нерелевантными.''')
code(prefix[split:]+'\ndataset = grouped_training(clean_history, expanded_training, "expanded_training")')
md('''## 11. Development: два режима и распределение запросов

Сравниваем новые тексты и знакомые тексты с удержанным контекстом. Во втором режиме
собственные context labels удалены, 90% целевых ID также удалены из истории; другие
контексты текста остаются. Это две симуляции тех же 2600 запросов, не 5200 независимых labels.
Доли режимов берутся только из текстового пересечения benchmark/train. Внутри режимов
веса согласуют длину запроса, наличие фильтров и локации с benchmark без его разметки.
Это приближение к распределению теста, не гарантированная оценка закрытой метрики.''')
body=selection[selection.index('def batched_new_scores'):]
body=body.replace('if path.exists():model=joblib.load(path)','if USE_CACHE and path.exists():model=joblib.load(path)')
split=body.index('base_groups=')
code(body[:split])
md('''## 12. Сравнение алгоритмов и объёма обучения

LightGBM LambdaRank и бинарный classifier; CatBoost YetiRank/NDCG и Logloss.
Каждый вариант проверен на 4000 и расширенных контекстах, при 200/400/800 деревьях.
RRF с reference: веса 25/50/75/100%. Reference также участвует как fallback.
Выбор только по development. Дополнительное условие: падение обычного macro Recall
новых текстов не превышает 0.5 п.п. При выборе новые 800 запросов не использовались.

По умолчанию Run All обучает уже выбранный LambdaRank: 200 деревьев, вес 75%.
Полный сравнительный эксперимент сохранён в development_experiments_v2.csv;
его можно повторить с AVITO_RUN_MODEL_SEARCH=1 до запуска ядра. Это не требуется
для воспроизведения финального ответа. Фиксированы гиперпараметры, а не ответы.''')
split2=body.index('test_records=')
search_code=body[split:split2]
fixed_code='''# Hyperparameters frozen from the development comparison, before the audit.
best = {"model": "lgb_rank_expanded", "family": "lgb_rank", "scale": "expanded",
        "trees": 200, "weight": .75}
path = CACHE / f"lgb_rank_expanded_200_{fingerprint}.joblib"
if USE_CACHE and path.exists():
    model = joblib.load(path)
else:
    order = np.argsort(dataset["group"], kind="stable")
    x, y, g = (dataset[key][order] for key in ["X", "y", "group"])
    _, sizes = np.unique(g, return_counts=True)
    model = lgb.LGBMRanker(n_estimators=200, num_leaves=31, learning_rate=.05,
        max_bin=127, min_child_samples=50, reg_lambda=10, random_state=SEED,
        n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True,
        lambdarank_truncation_level=55, label_gain=[0, 1])
    model.fit(x, y, group=sizes)
    save_cache(model, path)
    del x, y, g
    gc.collect()
model_paths = {best["model"]: str(path)}
scores = batched_new_scores(model, dev_features, best["family"], best["trees"])
seen_scores = batched_new_scores(model, seen_features, best["family"], best["trees"])
best.update(selection_metrics(combined_predict(dev_records, scores, reference, best["weight"]),
    combined_predict(seen_records, seen_scores, seen_reference, best["weight"])))
assert best["unknown_macro_recall50"] >= ref_metrics["unknown_macro_recall50"] - .005
(CACHE / "choice_v2.json").write_text(json.dumps({"selected": best,
    "model_paths": model_paths, "protocol": protocol, "config": EXPERIMENT_CONFIG,
    "known_query_fraction": known_fraction}, indent=2), encoding="utf-8")
print("Frozen development winner:", best, flush=True)
'''
code('if RUN_MODEL_SEARCH:\n'+textwrap.indent(search_code,'    ')+'\nelse:\n'+textwrap.indent(fixed_code,'    '))
md('''## 13. Зарезервированные 800 запросов: оценка один раз

Конфигурация уже зафиксирована. Метрика считается среди всего benchmark-корпуса,
включая positives, пропущенные candidate generation. Bootstrap разницы парный,
по запросам; audit не участвует в выборе параметров.''')
code(body[split2:])
code('''# Separate retrieval losses from losses while reducing the pool to 50.
analysis_rows = []
reference_predictions = direct_predict(test_records, test_reference)
for position, (row, truth, predicted, previous, record) in enumerate(zip(
        new_audit.itertuples(index=False), new_audit_labels, predictions,
        reference_predictions, test_records)):
    missed = sorted(set(truth) - set(predicted))
    pool_missed = sorted(set(truth) - set(record[0]))
    selection_missed = sorted(set(missed) - set(pool_missed))
    analysis_rows.append({"query": row.search_query, "location": row.search_location_id,
        "filters": row.search_infm_params_text, "reference_recall": float(a[position]),
        "selected_recall": float(b[position]), "pool_size": len(record[0]),
        "missed_before_pool": " ".join(ITEM_IDS[pool_missed]),
        "missed_at_selection": " ".join(ITEM_IDS[selection_missed]),
        "missed_titles": " | ".join(items.iloc[missed].item_title_raw.astype(str)),
        "top5_titles": " | ".join(items.iloc[predicted[:5]].item_title_raw.astype(str)),
        "improved": bool(b[position] > a[position]), "worsened": bool(b[position] < a[position])})
pd.DataFrame(analysis_rows).to_csv(CACHE / "audit_error_analysis.csv", index=False)
print("Saved detailed retrieval/selection losses and regressions")
''')
md('''## 14. Финальное переобучение и answer.csv

После всех оценок отложенные части возвращаются в финальное OOF-обучение.
Параметры зафиксированы; финальная модель не оценивается на уже использованных labels.
Для benchmark prior заменяется финальным prior v0.3.0. Новая модель смешивается
с ним с выбранным весом. Ответ проверяется по ID исходных файлов и повторно читается.''')
final=(root/'.development/ranking_final.py').read_text(encoding='utf-8')
code(defs(codes[16])[defs(codes[16]).index('def validate_answer'):]+ '\n'+ final)
md('''## 15. Интерпретация и воспроизводимость

Отправлять answer.csv. Код, описание, локальные веса/векторы и dependencies приложены.
Прежний реальный результат платформы v0.3.0 — Recall@50 0.858569. Новая версия
выбирается по локальным development-данным; её платформенная метрика пока неизвестна.
Режим known/unknown и согласование распределений — проверяемые гипотезы.
Разметка отражает пользовательский выбор, ограничена корпусным пересечением и неполна.

Использованы NumPy, pandas, SciPy, PyArrow, scikit-learn, Snowball, joblib,
PyTorch/Transformers, CatBoost и LightGBM. Модель E5 и алгоритмы открытые;
чужие решения задания не использовались. Источники:
- [Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-small)
- [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking)
- [LightGBM ranking API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
- [BM25](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)
''')
(root/'.development/notebook_source.py').write_text('\n'.join(parts),encoding='utf-8')
print('Generated self-contained ranking notebook source:',len(parts),'cells')
