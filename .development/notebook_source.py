# %% [markdown]
# # Кандидатогенерация услуг Авито: ranking v0.4.0
#
# BM25, символьный TF-IDF и локальная E5 создают пул; обучаемая модель отбирает 50 объявлений.
# Сравниваются ranking-loss и бинарная классификация, 4000 и расширенная выборка контекстов.
# В обучение добавлены трудные отрицательные примеры прежней модели. Выбор учитывает
# знакомые/новые тексты, длину запроса, фильтры и наличие локации в корпусе.
#
# Все вычисления локальны. Код решения целиком находится здесь; внешние API и разметка
# benchmark не используются. `search_category` не входит в scoring.
# Запуск: три Parquet в этой папке, зависимости из requirements.txt, локальные модели,
# **Restart Kernel → Run All**. Результат — answer.csv; артефакты — artifacts/ranking-v1.
# Прежние небольшие бустинги поставляются как замороженные priors. Их полная обучающая
# рецептура сохранена в Avito_v0.3.ipynb; новые модели обучаются в этом ноутбуке.

# %%
from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "train.parquet").exists() and (ROOT / "AvitoTest/train.parquet").exists():
    ROOT = ROOT / "AvitoTest"
if (ROOT / ".inspection_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".inspection_deps"))

if (ROOT / ".ranking_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".ranking_deps"))

import ast
import gc
import hashlib
import html
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

SEED = 260926
CONFIG = {
    "seed": SEED, "validation_texts": 1400, "development_fraction": 0.5,
    "cold_item_fraction": 0.9, "description_chars": 4000,
    "params_chars": 1800, "word_features": 240_000,
    "char_features": 220_000, "retrieval_pool": 500,
    "version": "ranking-v1",
}
RANKER_CONFIG = {
    "training_queries": 4000, "audit_queries": 600, "fresh_audit_queries": 600, "folds": 3,
    "hard_negatives": 100, "random_negatives": 60,
    "iterations": [120, 240], "leaf_options": [15, 31],
    "learning_rate": 0.08, "l2_regularization": 10.0,
    "blend_options": [0.5, 0.75, 1.0], "feature_version": 1,
}
SEMANTIC_CONFIG = {
    "model_id": "intfloat/multilingual-e5-small",
    "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
    "max_length": 192, "description_chars": 1200, "params_chars": 256, "batch_size": 128,
    "retrieval_pool": 500, "temperature": 0.05, "dtype": "float32",
}
CACHE = ROOT / "artifacts" / CONFIG["version"]
CACHE.mkdir(parents=True, exist_ok=True)
USE_CACHE = os.environ.get("AVITO_REBUILD_CACHE", "0") != "1"
threadpool_limits(limits=8)
np.random.seed(SEED)
print("Project:", ROOT)
print("Versions:", {"python": sys.version.split()[0], "pandas": pd.__version__,
                    "numpy": np.__version__, "sklearn": sklearn.__version__})
from catboost import CatBoostRanker, CatBoostClassifier
import lightgbm as lgb

# The search was completed on development before opening the 800-query audit.
# Default Run All trains the frozen winner; opt in to repeat the full comparison.
RUN_MODEL_SEARCH = os.getenv("AVITO_RUN_MODEL_SEARCH", "0") == "1"

# %% [markdown]
# ## 1. Данные и идентификаторы
#
# ID сохраняются строками, корпус сортируется по item_id. Тексты пропусков пустые,
# Decimal-цены/координаты переводятся в числовые массивы. Порядок документов фиксирован.

# %%
QUERY_COLS = ["search_query", "search_location_id", "search_is_delivery_search",
              "search_infm_params_text", "search_category"]
train = pd.read_parquet(ROOT / "train.parquet")
queries = pd.read_parquet(ROOT / "benchmark_queries.parquet")
items = pd.read_parquet(ROOT / "benchmark_items.parquet").sort_values("item_id").reset_index(drop=True)
for frame in [train, queries, items]:
    for column in frame.columns:
        if column.endswith("_raw") or column.endswith("_text") or column == "search_query":
            frame[column] = frame[column].fillna("").astype("string[pyarrow]")
    for column in ["item_price", "item_latitude", "item_longitude"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

assert len(queries) == queries.query_id.nunique()
assert queries.query_id.str.len().eq(16).all()
assert items.item_id.is_unique
assert items.item_id.str.fullmatch(r"[0-9a-f]{16}").all()
assert train.item_id.str.fullmatch(r"[0-9a-f]{16}").all()
assert set(QUERY_COLS).issubset(train.columns) and set(QUERY_COLS).issubset(queries.columns)
ITEM_IDS = items.item_id.to_numpy(dtype=str)
ITEM_TO_ROW = {item_id: index for index, item_id in enumerate(ITEM_IDS)}
print(pd.DataFrame([
    {"file": name, "rows": len(frame), "columns": frame.shape[1],
     "memory_MiB": round(frame.memory_usage(deep=True).sum() / 2**20, 1)}
    for name, frame in [("train", train), ("queries", queries), ("items", items)]
]).to_string(index=False))

# %% [markdown]
# ## 2. Нормализация и состав данных
#
# Одинаковая нормализация применяется к train и benchmark. Полный контекст включает
# текст, локацию, фильтры, доставку и категорию для сопоставления labels; категория
# не становится входом обучаемой модели. Повторные пары контекст/item_id удаляются.

# %%
def normalize_text(value):
    value = unicodedata.normalize("NFKC", html.unescape(str(value))).lower().replace("ё", "е")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()

for frame in [train, queries]:
    frame["query_norm"] = frame.search_query.map(normalize_text).astype("string[pyarrow]")
    # JSON сохраняет границы признаков: конкатенация без разделителей небезопасна.
    frame["context_key"] = [
        hashlib.sha256(json.dumps([str(v) for v in row], ensure_ascii=False).encode()).hexdigest()[:24]
        for row in frame[QUERY_COLS].itertuples(index=False, name=None)
    ]

eda = {
    "train_unique_query_texts": int(train.query_norm.nunique()),
    "train_unique_contexts": int(train.context_key.nunique()),
    "duplicate_context_item_pairs": int(train.duplicated(["context_key", "item_id"]).sum()),
    "benchmark_query_text_seen_fraction": float(queries.query_norm.isin(train.query_norm).mean()),
    "benchmark_item_seen_fraction": float(items.item_id.isin(train.item_id).mean()),
    "positive_same_location_fraction": float((train.search_location_id == train.item_location_id).mean()),
    "positive_same_category_fraction": float((train.search_category == train.item_category_id).mean()),
}
print(json.dumps(eda, ensure_ascii=False, indent=2))
print("\nДлины текстов корпуса, символы:")
print(items[["item_title_raw", "item_description_raw", "item_infm_params_text"]]
      .apply(lambda s: s.str.len().describe(percentiles=[.5, .95, .99])).round(1).to_string())

# %% [markdown]
# ## 3. Исторические выборки: теперь development
#
# Прежние 1400 validation и два среза по 600 запросов уже анализировались при разработке.
# Все 2600 теперь используются для выбора новой модели. Они исключены из истории
# нового обучения. Дополнительные 800 текстов будут зарезервированы до любого нового fit.
# Метки ограничены положительными объявлениями, присутствующими в доступном корпусе.

# %%
pairs = train.drop_duplicates(["context_key", "item_id"]).copy()
eligible = pairs[pairs.item_id.isin(ITEM_TO_ROW)].copy()
rng = np.random.default_rng(SEED)
text_pool = np.sort(eligible.query_norm.unique().astype(str))
selected_texts = rng.choice(text_pool, size=min(CONFIG["validation_texts"], len(text_pool)), replace=False)
selected = eligible[eligible.query_norm.isin(selected_texts)]
context_pool = selected[["query_norm", "context_key"]].drop_duplicates().sort_values(["query_norm", "context_key"])
chosen_keys = []
for _, group in context_pool.groupby("query_norm", sort=True):
    chosen_keys.append(group.context_key.iloc[int(rng.integers(len(group)))])
validation_pairs = selected[selected.context_key.isin(chosen_keys)]
validation = validation_pairs.drop_duplicates("context_key")[[*QUERY_COLS, "query_norm", "context_key"]].copy()
validation = validation.sort_values("context_key").reset_index(drop=True)
labels_by_key = validation_pairs.groupby("context_key").item_id.agg(lambda s: sorted(set(s)))
labels = [set(ITEM_TO_ROW[item_id] for item_id in labels_by_key[key]) for key in validation.context_key]
val_positive_ids = np.sort(validation_pairs.item_id.unique().astype(str))
cold_item_ids = set(rng.choice(val_positive_ids, size=int(CONFIG["cold_item_fraction"] * len(val_positive_ids)), replace=False))
fit_pairs = pairs[~pairs.query_norm.isin(selected_texts) & ~pairs.item_id.isin(cold_item_ids)].copy()
assert not set(validation.query_norm) & set(fit_pairs.query_norm)
assert not cold_item_ids & set(fit_pairs.item_id)
assert all(labels)
split_order = rng.permutation(len(validation))
dev_indices = np.sort(split_order[:int(len(validation) * CONFIG["development_fraction"])])
holdout_indices = np.sort(split_order[int(len(validation) * CONFIG["development_fraction"]):])
validation["split"] = "holdout"
validation.loc[dev_indices, "split"] = "development"
validation["relevant_item_ids"] = [" ".join(labels_by_key[key]) for key in validation.context_key]
validation.to_parquet(CACHE / "validation.parquet", index=False)
print("History pairs:", len(fit_pairs), "; validation queries:", len(validation))
print("Development:", len(dev_indices), "; holdout:", len(holdout_indices))
print("Mean positives:", round(np.mean([len(x) for x in labels]), 3))
print("Actual unseen positive item fraction:", round(1 - validation_pairs.item_id.isin(fit_pairs.item_id).mean(), 4))
# Полные тексты train больше не нужны; сохраняем компактную историю для обучения.
HISTORY_COLS = [*QUERY_COLS, "query_norm", "context_key", "item_id", "item_location_id", "item_microcat_id"]
history_all = pairs[HISTORY_COLS].copy()
history_fit = fit_pairs[HISTORY_COLS].copy()
del train, pairs, fit_pairs, eligible, selected, validation_pairs
gc.collect()

# %% [markdown]
# ## 4. Лексические индексы
#
# BM25 заголовка и полного текста, русский Snowball stemmer, символьный TF-IDF.
# Индексы строятся без labels по всем объявлениям корпуса. Кеш необязателен;
# AVITO_REBUILD_CACHE=1 перестраивает производные данные и новые модели.

# %%
stemmer = snowballstemmer.stemmer("russian")
STOP_WORDS = set("и в во на с со к ко по из за от до для у о об а но или это как что я мы вы он она они не без при под над".split())

@lru_cache(maxsize=350_000)
def stem_word(word):
    return stemmer.stemWord(word)

def tokenize_words(text):
    return [stem_word(w) for w in re.findall(r"[a-zа-я0-9]+", normalize_text(text))
            if len(w) > 1 and w not in STOP_WORDS]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

input_hashes = {name: sha256_file(ROOT / name) for name in
                ["train.parquet", "benchmark_items.parquet", "benchmark_queries.parquet"]}
notebook_document = json.loads((ROOT / "Avito.ipynb").read_text(encoding="utf-8"))
code_source = "\n".join("".join(cell["source"]) for cell in notebook_document["cells"] if cell["cell_type"] == "code")
code_hash = hashlib.sha256(code_source.encode("utf-8")).hexdigest()
fingerprint = hashlib.sha256(json.dumps({"inputs": input_hashes, "config": CONFIG,
                                        "ranker": RANKER_CONFIG, "semantic": SEMANTIC_CONFIG,
                                        "code": code_hash}, sort_keys=True).encode()).hexdigest()[:16]
index_path = CACHE / f"lexical_{fingerprint}.joblib"

def save_cache(value, path):
    # Atomic publication: an interrupted write cannot become a valid cache file.
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=0)
    temporary.replace(path)

class BM25Index:
    def __init__(self, texts, max_features, k1=1.2, b=0.75):
        self.vectorizer = CountVectorizer(tokenizer=tokenize_words, token_pattern=None,
                                          lowercase=False, dtype=np.float32,
                                          max_features=max_features)
        counts = self.vectorizer.fit_transform(texts).tocsr()
        n = counts.shape[0]
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float32)
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1))
        row_norm = np.repeat(norm, np.diff(counts.indptr))
        counts.data = (counts.data * (k1 + 1) / (counts.data + row_norm) * self.idf[counts.indices]).astype(np.float32)
        self.matrix = counts
        self.transpose = counts.T.tocsr()

    def query_matrix(self, texts):
        matrix = self.vectorizer.transform(texts).tocsr()
        matrix.data[:] = 1  # repeated query words do not increase importance
        return matrix

started = time.perf_counter()
if USE_CACHE and index_path.exists():
    lexical = joblib.load(index_path)
    print("Loaded lexical cache")
else:
    titles = items.item_title_raw.map(normalize_text).tolist()
    bodies = (items.item_title_raw + " " + items.item_infm_params_text.str.slice(0, CONFIG["params_chars"])
              + " " + items.item_description_raw.str.slice(0, CONFIG["description_chars"])).tolist()
    print("Building title BM25...", flush=True)
    title_index = BM25Index(titles, CONFIG["word_features"])
    print("Building body BM25...", flush=True)
    body_index = BM25Index(bodies, CONFIG["word_features"])
    del bodies
    print("Building character title index...", flush=True)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                     max_features=CONFIG["char_features"], sublinear_tf=True,
                                     dtype=np.float32)
    char_matrix = char_vectorizer.fit_transform(titles).tocsr()
    lexical = {"title": title_index, "body": body_index, "char_vectorizer": char_vectorizer,
               "char_transpose": char_matrix.T.tocsr()}
    save_cache(lexical, index_path)
    del titles, char_matrix, title_index, body_index, char_vectorizer
    gc.collect()
print("Index preparation seconds:", round(time.perf_counter() - started, 1))

# %% [markdown]
# ## 5. География и перенос намерения из истории
#
# Точное совпадение города, расстояние и обученные переходы локаций — мягкие признаки.
# Похожие тексты истории задают распределение подкатегорий. Целевые взаимодействия
# оценочных запросов удаляются из истории до построения их признаков.

# %%
ITEM_LOCS = items.item_location_id.to_numpy()
ITEM_MICROS = items.item_microcat_id.to_numpy()
MICROS = np.sort(items.item_microcat_id.unique())
MICRO_TO_COL = {int(value): i for i, value in enumerate(MICROS)}
ITEM_MICRO_COLS = np.array([MICRO_TO_COL[int(x)] for x in ITEM_MICROS])
centers = items.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
item_lat = np.radians(items.item_latitude.to_numpy())
item_lon = np.radians(items.item_longitude.to_numpy())

def stable_topk(scores, k=50):
    """Descending score, then ascending item row (= ascending item_id)."""
    scores = np.nan_to_num(np.asarray(scores), nan=-np.inf)
    k = min(k, len(scores))
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:k - len(above)]
    chosen = np.concatenate([above, tied])
    return chosen[np.lexsort((chosen, -scores[chosen]))]

class HistorySignals:
    def __init__(self, history):
        transitions = history.groupby(["search_location_id", "item_location_id"]).size()
        self.transitions = {}
        for search_loc, counts in transitions.groupby(level=0):
            # Shrink small-sample transitions; do not hard-filter to observed cities.
            maximum = max(float(counts.max()), 1)
            strength = min(1.0, float(counts.sum()) / 20)
            self.transitions[int(search_loc)] = {
                int(pair[1]): float(np.sqrt(count / maximum) * strength)
                for pair, count in counts.items()
            }
        texts = np.sort(history.query_norm.unique().astype(str))
        self.texts = texts
        self.text_to_row = {text: i for i, text in enumerate(texts)}
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         min_df=2, max_features=180_000, sublinear_tf=True,
                                         dtype=np.float32)
        self.text_matrix_t = self.vectorizer.fit_transform(texts).T.tocsr()
        valid = history[history.item_microcat_id.isin(MICRO_TO_COL)]
        counts = valid.groupby(["query_norm", "item_microcat_id"]).size()
        rows = [self.text_to_row[text] for text, _ in counts.index]
        columns = [MICRO_TO_COL[int(micro)] for _, micro in counts.index]
        self.micro_matrix = sp.csr_matrix((counts.to_numpy(dtype=np.float32), (rows, columns)),
                                          shape=(len(texts), len(MICROS)))
        sums = np.asarray(self.micro_matrix.sum(axis=1)).ravel()
        self.micro_matrix = sp.diags(1 / np.maximum(sums, 1)) @ self.micro_matrix

    def micro_predictions(self, query_texts):
        query_matrix = self.vectorizer.transform(query_texts)
        outputs = []
        for row in range(query_matrix.shape[0]):
            similarities = (query_matrix[row] @ self.text_matrix_t).toarray().ravel()
            nearest = stable_topk(similarities, 20)
            weights = np.maximum(similarities[nearest] - 0.2, 0) ** 3
            # Exact known query carries more evidence than several weak neighbours.
            exact = self.text_to_row.get(query_texts[row])
            if exact is not None:
                nearest = np.array([exact])
                weights = np.array([1.0], dtype=np.float32)
            probabilities = np.asarray(self.micro_matrix[nearest].T @ weights).ravel()
            if probabilities.max() > 0:
                probabilities /= probabilities.max()
            else:
                probabilities[:] = 1
            outputs.append(probabilities.astype(np.float32))
        return np.stack(outputs)

    def geography(self, search_location):
        result = np.zeros(len(items), dtype=np.float32)
        if search_location in centers.index:
            lat, lon = np.radians(centers.loc[search_location].to_numpy(dtype=float))
            hav = np.sin((item_lat - lat) / 2)**2 + np.cos(lat) * np.cos(item_lat) * np.sin((item_lon - lon) / 2)**2
            distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
            result = np.nan_to_num(0.7 * np.exp(-distance / 40), nan=0).astype(np.float32)
        mapping = self.transitions.get(int(search_location), {})
        if mapping:
            unique_locs, inverse = np.unique(ITEM_LOCS, return_inverse=True)
            transition = np.array([mapping.get(int(loc), 0) for loc in unique_locs], dtype=np.float32)[inverse]
            result = np.maximum(result, transition)
        result[ITEM_LOCS == search_location] = 1
        if result.max() == 0:
            result[:] = 1
        return result

def select_query_contexts(history, count, seed):
    eligible_pairs = history[history.item_id.isin(ITEM_TO_ROW)]
    local_rng = np.random.default_rng(seed)
    texts = np.sort(eligible_pairs.query_norm.unique().astype(str))
    chosen_texts = local_rng.choice(texts, min(count, len(texts)), replace=False)
    contexts = eligible_pairs[eligible_pairs.query_norm.isin(chosen_texts)].drop_duplicates("context_key")
    chosen = []
    for _, group in contexts.sort_values("context_key").groupby("query_norm", sort=True):
        chosen.append(group.iloc[int(local_rng.integers(len(group)))])
    return pd.DataFrame(chosen)[[*QUERY_COLS, "query_norm", "context_key"]].sort_values("context_key").reset_index(drop=True)

def query_labels(history, query_frame):
    pairs = history[history.context_key.isin(query_frame.context_key) & history.item_id.isin(ITEM_TO_ROW)]
    grouped = pairs.groupby("context_key").item_id.agg(lambda s: set(ITEM_TO_ROW[x] for x in s))
    return [grouped.get(key, set()) for key in query_frame.context_key]

def purge_history(history, query_frame, relevant, seed):
    positives = np.array(sorted({ITEM_IDS[i] for truth in relevant for i in truth}))
    local_rng = np.random.default_rng(seed)
    cold_ids = set(local_rng.choice(positives, int(CONFIG["cold_item_fraction"] * len(positives)), replace=False))
    clean = history[~history.query_norm.isin(query_frame.query_norm) & ~history.item_id.isin(cold_ids)].copy()
    assert not set(query_frame.query_norm) & set(clean.query_norm)
    assert not cold_ids & set(clean.item_id)
    return clean, cold_ids

audit_queries = select_query_contexts(history_fit, RANKER_CONFIG["audit_queries"], SEED + 101)
audit_labels = query_labels(history_fit, audit_queries)
ranker_history, audit_cold_ids = purge_history(history_fit, audit_queries, audit_labels, SEED + 102)
original_training_queries = select_query_contexts(ranker_history, RANKER_CONFIG["training_queries"], SEED + 103)
# These 600 texts were not direct training queries in the already inspected v0.2
# model. Remove their labels from all history before fitting the new comparison.
fresh_source = ranker_history[~ranker_history.query_norm.isin(original_training_queries.query_norm)]
fresh_queries = select_query_contexts(fresh_source, RANKER_CONFIG["fresh_audit_queries"], SEED + 501)
fresh_labels = query_labels(ranker_history, fresh_queries)
ranker_history, fresh_cold_ids = purge_history(ranker_history, fresh_queries, fresh_labels, SEED + 502)
training_queries = select_query_contexts(ranker_history, RANKER_CONFIG["training_queries"], SEED + 103)
assert all(fresh_labels)
assert not set(fresh_queries.query_norm) & set(original_training_queries.query_norm)
assert not set(fresh_queries.query_norm) & set(training_queries.query_norm)
assert not set(fresh_queries.query_norm) & set(validation.query_norm)
assert not set(fresh_queries.query_norm) & set(audit_queries.query_norm)
assert not set(fresh_queries.query_norm) & set(ranker_history.query_norm)
assert not fresh_cold_ids & set(ranker_history.item_id)
fresh_protocol = {"fresh_queries": len(fresh_queries), "development_queries": len(validation) + len(audit_queries),
    "training_queries": len(training_queries), "query_text_overlap": 0, "purged_item_overlap": 0,
    "old_training_query_text_overlap": 0, "purged_positive_items": len(fresh_cold_ids),
    "used_for_model_selection": False}
fresh_protocol["role"] = "historical_v0.3_slice_now_development"
fresh_protocol["used_for_model_selection"] = True
(CACHE / "prior_split_protocol.json").write_text(json.dumps(fresh_protocol, indent=2), encoding="utf-8")
del fresh_source

started = time.perf_counter()
history_model = HistorySignals(ranker_history)
print("History signals prepared in", round(time.perf_counter() - started, 1), "seconds")

# %% [markdown]
# ## 6. Исходные кандидаты и 38 признаков
#
# Лексические источники сохраняются; позднее добавляются E5 cosine и cosine с географией.
# Обучаемый отбор использует текстовые оценки/ранги/покрытие, географию, подкатегории,
# рейтинг, отзывы, цену и флаги связи. Сырые ID и search_category в модели отсутствуют.

# %%
def normalized_scores(scores):
    maximum = float(scores.max())
    return scores / maximum if maximum > 0 else scores

def retrieve_features(query_frame, history_model, progress_every=200):
    texts = query_frame.query_norm.astype(str).tolist()
    title_queries = lexical["title"].query_matrix(texts)
    body_queries = lexical["body"].query_matrix(texts)
    filter_queries = lexical["body"].query_matrix(query_frame.search_infm_params_text.astype(str).tolist())
    char_queries = lexical["char_vectorizer"].transform(texts)
    micro_predictions = history_model.micro_predictions(texts)
    outputs = []
    geo_cache = {}
    started = time.perf_counter()
    for i, query in enumerate(query_frame.itertuples(index=False)):
        title = normalized_scores((title_queries[i] @ lexical["title"].transpose).toarray().ravel())
        body = normalized_scores((body_queries[i] @ lexical["body"].transpose).toarray().ravel())
        chars = normalized_scores((char_queries[i] @ lexical["char_transpose"]).toarray().ravel())
        filters = normalized_scores((filter_queries[i] @ lexical["body"].transpose).toarray().ravel())
        loc = int(query.search_location_id)
        if loc not in geo_cache:
            geo_cache[loc] = history_model.geography(loc)
        geo = geo_cache[loc]
        micro = micro_predictions[i][ITEM_MICRO_COLS]
        sources = [title, body, chars]
        candidate_parts = []
        for source in sources:
            candidate_parts.append(stable_topk(source, CONFIG["retrieval_pool"]))
            candidate_parts.append(stable_topk(source * (0.03 + 0.97 * geo), CONFIG["retrieval_pool"]))
        mixed = (0.5 * title + 0.3 * body + 0.2 * chars)
        candidate_parts.append(stable_topk(mixed * (0.03 + 0.97 * geo) * (0.2 + 0.8 * micro), CONFIG["retrieval_pool"]))
        candidates = np.unique(np.concatenate(candidate_parts))
        features = np.column_stack([title[candidates], body[candidates], chars[candidates],
                                    filters[candidates], geo[candidates], micro[candidates]]).astype(np.float32)
        outputs.append((candidates, features))
        if (i + 1) % progress_every == 0 or i + 1 == len(query_frame):
            print(f"Retrieved {i + 1}/{len(query_frame)} queries; {time.perf_counter() - started:.1f}s", flush=True)
    return outputs

def score_candidates(features, config):
    title, body, chars, filters, geo, micro = features.T
    score = config["title"] * title + config["body"] * body + config["char"] * chars + config["filter"] * filters
    score *= (0.03 + 0.97 * geo) ** config["geo"]
    score *= (0.2 + 0.8 * micro) ** config["micro"]
    return score

def predict_from_features(records, config, k=50):
    return [candidates[stable_topk(score_candidates(features, config), k)] for candidates, features in records]

def per_query_recall(predictions, relevant):
    assert len(predictions) == len(relevant)
    return np.array([len(set(map(int, predicted)) & truth) / len(truth)
                     for predicted, truth in zip(predictions, relevant)], dtype=float)
best_config = {'name': 'hybrid_19', 'title': 0.3, 'body': 0.5, 'char': 0.2, 'filter': 0.03, 'geo': 0.5, 'micro': 0.5}

# %%
RANK_FEATURE_NAMES = [
    "title_bm25", "body_bm25", "char_tfidf", "filter_bm25", "geo_compatibility", "micro_compatibility",
    "baseline_score", "log_baseline_rank", "log_title_rank", "log_body_rank", "log_char_rank",
    "same_location", "log_distance_km", "title_token_coverage", "body_token_coverage",
    "title_idf_coverage", "body_idf_coverage", "query_in_title", "query_equals_title",
    "query_characters", "query_tokens", "log_filter_characters", "rating_required", "rating_gap",
    "rating", "log_reviews", "log_price", "phone_hidden", "messages_forbidden",
    "log_title_characters", "log_description_characters", "log_params_characters",
]
assert not any("category" in name or name.endswith("_id") for name in RANK_FEATURE_NAMES)
normalized_titles = items.item_title_raw.map(normalize_text).to_numpy(dtype=str)
item_quality = np.column_stack([
    items.item_rating.to_numpy(dtype=np.float32),
    np.log1p(items.item_rating_reviews_count.clip(lower=0).to_numpy(dtype=np.float32)),
    np.log1p(items.item_price.clip(lower=0).to_numpy(dtype=np.float32)),
    items.item_is_phone_hidden.to_numpy(dtype=np.float32),
    items.item_is_message_forbidden.to_numpy(dtype=np.float32),
    np.log1p(items.item_title_raw.str.len().to_numpy(dtype=np.float32)),
    np.log1p(items.item_description_raw.str.len().to_numpy(dtype=np.float32)),
    np.log1p(items.item_infm_params_text.str.len().to_numpy(dtype=np.float32)),
]).astype(np.float32)

def token_coverage(index, candidates, terms):
    vocabulary = index.vectorizer.vocabulary_
    indices = [vocabulary[term] for term in terms if term in vocabulary]
    if not indices:
        return np.zeros((len(candidates), 2), dtype=np.float32)
    present = (index.matrix[candidates][:, indices].toarray() > 0).astype(np.float32)
    idf = index.idf[indices]
    unknown = len(terms) - len(indices)
    denominator = float(idf.sum()) + unknown * float(index.idf.max())
    return np.column_stack([present.sum(axis=1) / max(len(terms), 1),
                            present @ idf / max(denominator, 1e-6)])

def rank_features(query, record):
    candidates, base = record
    n = len(candidates)
    baseline = score_candidates(base, best_config)
    ranks = np.column_stack([np.log1p(rankdata(-values, method="min")) for values in
                             [baseline, base[:, 0], base[:, 1], base[:, 2]]])
    same_location = (ITEM_LOCS[candidates] == int(query.search_location_id)).astype(np.float32)
    distance = np.full(n, np.nan, dtype=np.float32)
    if query.search_location_id in centers.index:
        lat, lon = np.radians(centers.loc[query.search_location_id].to_numpy(dtype=float))
        hav = np.sin((item_lat[candidates] - lat) / 2)**2 + np.cos(lat) * np.cos(item_lat[candidates]) * np.sin((item_lon[candidates] - lon) / 2)**2
        distance = np.log1p(6371 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1))))
    terms = sorted(set(tokenize_words(query.query_norm)))
    title_coverage = token_coverage(lexical["title"], candidates, terms)
    body_coverage = token_coverage(lexical["body"], candidates, terms)
    title_texts = normalized_titles[candidates]
    substring = np.array([query.query_norm in text for text in title_texts], dtype=np.float32)
    exact = (title_texts == query.query_norm).astype(np.float32)
    threshold_match = re.search(r"рейтинг[^\d]{0,40}([0-5])", normalize_text(query.search_infm_params_text))
    required_rating = float(threshold_match.group(1)) if threshold_match else np.nan
    query_constants = np.tile([len(query.query_norm), len(terms), np.log1p(len(query.search_infm_params_text)), required_rating], (n, 1))
    result = np.column_stack([
        base, baseline, ranks, same_location, distance,
        title_coverage[:, 0], body_coverage[:, 0], title_coverage[:, 1], body_coverage[:, 1],
        substring, exact, query_constants, item_quality[candidates, 0] - required_rating,
        item_quality[candidates],
    ]).astype(np.float32)
    assert result.shape == (n, 32)
    assert not np.isinf(result).any()
    return result

def hard_negative_sample(candidates, base, truth, local_rng):
    """Keep retrieved positives, hard negatives and a random tail; never inject labels."""
    target = np.isin(candidates, list(truth))
    positive = np.flatnonzero(target)
    negative = np.flatnonzero(~target)
    if not len(positive) or not len(negative):
        return np.array([], dtype=int), target
    ordered = negative[stable_topk(score_candidates(base[negative], best_config), len(negative))]
    hard = ordered[:RANKER_CONFIG["hard_negatives"]]
    tail = ordered[len(hard):]
    random_tail = local_rng.choice(tail, min(len(tail), RANKER_CONFIG["random_negatives"]), replace=False)
    selected = np.sort(np.concatenate([positive, hard, random_tail]))
    assert not set(candidates[selected[~target[selected]]]) & truth
    return selected, target


def build_rank_feature_records(query_frame, records, stage):
    path = CACHE / f"{stage}_rank_features_{fingerprint}.joblib"
    if USE_CACHE and path.exists():
        return joblib.load(path)
    output = [rank_features(query, record) for query, record in zip(query_frame.itertuples(index=False), records)]
    save_cache(output, path)
    return output

def model_scores(model, feature_records):
    sizes = [len(x) for x in feature_records]
    # Larger batches avoid repeated OpenMP startup for every individual query.
    output = []
    for start in range(0, len(feature_records), 64):
        block = feature_records[start:start + 64]
        scores = model.decision_function(np.concatenate(block))
        output.extend(np.split(scores, np.cumsum(sizes[start:start + len(block)])[:-1]))
    return output

def blend_predictions(records, learned_scores, alpha, k=50):
    if alpha == 0:
        return predict_from_features(records, best_config, k)
    predictions = []
    for (candidates, features), learned in zip(records, learned_scores):
        if alpha == 1:
            score = learned
        else:
            baseline = score_candidates(features, best_config)
            score = alpha / (60 + rankdata(-learned, method="min")) + (1 - alpha) / (60 + rankdata(-baseline, method="min"))
        predictions.append(candidates[stable_topk(score, k)])
    return predictions

# %% [markdown]
# ## 7. Локальный семантический энкодер
#
# intfloat/multilingual-e5-small, MIT, фиксированная ревизия и контрольные суммы.
# query:/passage:, attention-mask mean pooling, L2-нормировка, максимум 192 токена.
# Энкодер не дообучается; эмбеддинги не используют поведенческие labels. Точный поиск
# по корпусу выполняется на CPU с float64-накоплением. GPU нужен для кодирования
# при отсутствии приложенных векторов.

# %%
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if (ROOT / ".semantic_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".semantic_deps"))
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(8)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = ROOT / "models" / "multilingual-e5-small"
if not (MODEL_DIR / "manifest.json").exists():
    raise FileNotFoundError("Prepare models/multilingual-e5-small before offline execution; see README")
model_manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
assert model_manifest["revision"] == SEMANTIC_CONFIG["revision"]
for name, expected in model_manifest["sha256"].items():
    assert (MODEL_DIR / name).exists(), f"Missing local model file: {name}"
    assert sha256_file(MODEL_DIR / name) == expected, f"Modified model file: {name}"

def save_array(array, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)

class SemanticIndex:
    def __init__(self):
        self.tokenizer = None
        self.encoder = None

    def encode(self, texts, prefix):
        if self.encoder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
            self.encoder = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True,
                                                     attn_implementation="eager").to(DEVICE).eval()
        # A stable length order reduces padding without changing document IDs.
        order = np.argsort(np.array([len(text) for text in texts]), kind="stable")
        output = np.empty((len(texts), 384), dtype=np.float32)
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(order), SEMANTIC_CONFIG["batch_size"]):
                positions = order[start:start + SEMANTIC_CONFIG["batch_size"]]
                batch = self.tokenizer([prefix + texts[i] for i in positions], padding=True,
                    truncation=True, max_length=SEMANTIC_CONFIG["max_length"], return_tensors="pt")
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                hidden = self.encoder(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                embeddings = torch.nn.functional.normalize(pooled, p=2, dim=1)
                output[positions] = embeddings.cpu().numpy()
                completed = min(start + len(positions), len(order))
                if completed % (SEMANTIC_CONFIG["batch_size"] * 50) == 0 or completed == len(order):
                    elapsed = time.perf_counter() - started
                    print(f"E5 {prefix.strip()} {completed}/{len(order)}; {elapsed:.1f}s; {completed / max(elapsed, .01):.0f} texts/s", flush=True)
        assert np.isfinite(output).all()
        assert np.allclose(np.linalg.norm(output, axis=1), 1, atol=2e-5)
        return output

    def prepare(self, query_texts):
        documents = (items.item_title_raw + ". " + items.item_infm_params_text.str.slice(0, SEMANTIC_CONFIG["params_chars"])
                     + ". " + items.item_description_raw.str.slice(0, SEMANTIC_CONFIG["description_chars"]))
        # Encoder caches depend on text, model, input order and encoding parameters,
        # rather than the ranker code. AVITO_REBUILD_CACHE=1 still rebuilds everything.
        payload = {"model": model_manifest, "encoding": SEMANTIC_CONFIG, "items": input_hashes["benchmark_items.parquet"]}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        self.item_path = CACHE / f"e5_items_{key}.npy"
        if USE_CACHE and self.item_path.exists():
            self.items = np.load(self.item_path, allow_pickle=False)
            print("Loaded E5 document embeddings")
        else:
            self.items = self.encode(documents.astype(str).tolist(), "passage: ")
            save_array(self.items, self.item_path)
        texts = sorted(set(map(str, query_texts)))
        text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()[:16]
        self.query_path = CACHE / f"e5_queries_{key}_{text_hash}.npy"
        if USE_CACHE and self.query_path.exists():
            self.queries = np.load(self.query_path, allow_pickle=False)
        else:
            self.queries = self.encode(texts, "query: ")
            save_array(self.queries, self.query_path)
        self.query_to_row = {text: i for i, text in enumerate(texts)}
        assert self.items.shape == (len(items), 384)
        assert self.queries.shape == (len(texts), 384)
        del self.encoder, self.tokenizer
        self.encoder = self.tokenizer = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        self.item_transpose = self.items.astype(np.float64).T

    def scores(self, texts):
        # Also support new queries outside the notebook's prepared evaluation set.
        missing = sorted(set(map(str, texts)) - self.query_to_row.keys())
        if missing:
            additional = self.encode(missing, "query: ")
            offset = len(self.queries)
            self.queries = np.concatenate([self.queries, additional])
            self.query_to_row.update({text: offset + i for i, text in enumerate(missing)})
        vectors = self.queries[[self.query_to_row[str(text)] for text in texts]]
        return (vectors.astype(np.float64) @ self.item_transpose).astype(np.float32)

# %%
legacy_retrieve_features = retrieve_features
legacy_rank_features = rank_features
legacy_score_candidates = score_candidates
legacy_negative_sample = hard_negative_sample
LEGACY_FEATURE_NAMES = list(RANK_FEATURE_NAMES)
SEMANTIC_FEATURE_NAMES = ["e5_cosine", "e5_geo_affinity", "from_lexical_pool", "from_semantic_pool", "log_e5_rank", "log_e5_geo_rank"]
RANK_FEATURE_NAMES = LEGACY_FEATURE_NAMES + SEMANTIC_FEATURE_NAMES
def semantic_affinity(cosine, geography):
    return np.exp((cosine - 1) / SEMANTIC_CONFIG["temperature"]) * (0.03 + 0.97 * geography)

def retrieve_semantic_features(query_frame, history_model, progress_every=200):
    texts = query_frame.query_norm.astype(str).tolist()
    title_queries = lexical["title"].query_matrix(texts)
    body_queries = lexical["body"].query_matrix(texts)
    filter_queries = lexical["body"].query_matrix(query_frame.search_infm_params_text.astype(str).tolist())
    char_queries = lexical["char_vectorizer"].transform(texts)
    micro_predictions = history_model.micro_predictions(texts)
    outputs, geo_cache = [], {}
    started = time.perf_counter()
    dense_block = None
    for i, query in enumerate(query_frame.itertuples(index=False)):
        if i % 64 == 0:
            dense_block = semantic_index.scores(texts[i:i + 64])
        cosine = dense_block[i % 64]
        title = normalized_scores((title_queries[i] @ lexical["title"].transpose).toarray().ravel())
        body = normalized_scores((body_queries[i] @ lexical["body"].transpose).toarray().ravel())
        chars = normalized_scores((char_queries[i] @ lexical["char_transpose"]).toarray().ravel())
        filters = normalized_scores((filter_queries[i] @ lexical["body"].transpose).toarray().ravel())
        loc = int(query.search_location_id)
        if loc not in geo_cache:
            geo_cache[loc] = history_model.geography(loc)
        geo = geo_cache[loc]
        micro = micro_predictions[i][ITEM_MICRO_COLS]
        candidate_parts = []
        for source in [title, body, chars]:
            candidate_parts.append(stable_topk(source, CONFIG["retrieval_pool"]))
            candidate_parts.append(stable_topk(source * (0.03 + 0.97 * geo), CONFIG["retrieval_pool"]))
        mixed = 0.5 * title + 0.3 * body + 0.2 * chars
        candidate_parts.append(stable_topk(mixed * (0.03 + 0.97 * geo) * (0.2 + 0.8 * micro), CONFIG["retrieval_pool"]))
        lexical_candidates = np.unique(np.concatenate(candidate_parts))
        affinity = semantic_affinity(cosine, geo)
        semantic_candidates = np.union1d(stable_topk(cosine, SEMANTIC_CONFIG["retrieval_pool"]),
                                          stable_topk(affinity, SEMANTIC_CONFIG["retrieval_pool"]))
        candidates = np.union1d(lexical_candidates, semantic_candidates)
        features = np.column_stack([title[candidates], body[candidates], chars[candidates],
            filters[candidates], geo[candidates], micro[candidates], cosine[candidates], affinity[candidates],
            np.isin(candidates, lexical_candidates), np.isin(candidates, semantic_candidates)]).astype(np.float32)
        outputs.append((candidates, features))
        if (i + 1) % progress_every == 0 or i + 1 == len(query_frame):
            print(f"Hybrid retrieved {i + 1}/{len(query_frame)}; {time.perf_counter() - started:.1f}s", flush=True)
    return outputs

def score_lexical_columns(features, config):
    return legacy_score_candidates(features[:, :6], config)

def rank_semantic_features(query, record):
    candidates, features = record
    original = legacy_rank_features(query, (candidates, features[:, :6]))
    semantic = features[:, 6:10]
    ranks = np.column_stack([np.log1p(rankdata(-values, method="min"))
                             for values in [semantic[:, 0], semantic[:, 1]]])
    result = np.column_stack([original, semantic, ranks]).astype(np.float32)
    assert result.shape == (len(candidates), len(RANK_FEATURE_NAMES))
    return result

def semantic_negative_sample(candidates, features, truth, local_rng):
    target = np.isin(candidates, list(truth))
    positive, negative = np.flatnonzero(target), np.flatnonzero(~target)
    if not len(positive) or not len(negative):
        return np.array([], dtype=int), target
    half = RANKER_CONFIG["hard_negatives"] // 2
    lexical_hard = negative[stable_topk(score_candidates(features[negative], best_config), half)]
    semantic_hard = negative[stable_topk(features[negative, 7], half)]
    hard = np.union1d(lexical_hard, semantic_hard)
    tail = np.setdiff1d(negative, hard)
    random_tail = local_rng.choice(tail, min(len(tail), RANKER_CONFIG["random_negatives"]), replace=False)
    selected = np.sort(np.concatenate([positive, hard, random_tail]))
    assert not set(candidates[selected[~target[selected]]]) & truth
    return selected, target

def cached_hybrid_records(query_frame, history, stage):
    path = CACHE / f"{stage}_hybrid_{fingerprint}.joblib"
    if USE_CACHE and path.exists():
        return joblib.load(path)
    # Always measure the expanded pool, even if selection falls back to lexical.
    records = retrieve_semantic_features(query_frame, history)
    save_cache(records, path)
    return records
retrieve_features = retrieve_semantic_features
rank_features = rank_semantic_features
score_candidates = score_lexical_columns
hard_negative_sample = semantic_negative_sample

# %% [markdown]
# ## 8. Замороженные priors v0.3.0
#
# Четыре небольших HGB-модели обучены на предоставленном train, не на разметке benchmark.
# Оценочные модели на 4000 текстах используются для reference и поиска трудных negatives.
# Две финальные модели на 6600 текстах используются только при итоговом benchmark inference.
# Вес новой семантической модели в прежнем ансамбле — 25%, прежнего отбора — 75%.
# Файлы и их происхождение проверяются по SHA-256. Рецептура — Avito_v0.3.ipynb.

# %%
PRIORS_DIR = ROOT / "models" / "retrieval-priors"
prior_manifest = json.loads((PRIORS_DIR / "manifest.json").read_text(encoding="utf-8"))
assert prior_manifest["input_sha256"] == input_hashes
for name, expected in prior_manifest["sha256"].items():
    assert sha256_file(PRIORS_DIR / name) == expected, f"Modified frozen prior: {name}"
prior_legacy = joblib.load(PRIORS_DIR / "evaluation_legacy.joblib")
prior_semantic = joblib.load(PRIORS_DIR / "evaluation_semantic.joblib")

# %% [markdown]
# ## 9. Новое разбиение и расширение контекстов
#
# Новые 800 текстов не входят в прежние прямые обучающие запросы; их положительные ID
# также не были прямыми positives оценочных priors. Тексты и 90% целевых ID удаляются
# из истории до нового обучения. До четырёх дополнительных контекстов текста расширяют
# географию/фильтры; прежний прямой обучающий контекст сохраняется отдельно.
# Фактический объём определяется доступными положительными объявлениями корпуса;
# 20000 — верхняя граница, а не обещание наличия 20000 размеченных примеров.

# %%
EXPERIMENT_CONFIG = {'contexts': 20000, 'contexts_per_text': 4, 'audit_texts': 800,
    'hard_source': 40, 'hard_model': 80, 'random_tail': 64, 'folds': 3, 'seed': SEED + 701}
# Revision covers the copied numerical retrieval/feature/sampling routines.
# Changing those routines requires incrementing data_revision; full rebuild ignores caches.
fingerprint = hashlib.sha256(json.dumps({"inputs": input_hashes,
    "config": EXPERIMENT_CONFIG, "baseline_config": CONFIG, "semantic": SEMANTIC_CONFIG,
    "model": model_manifest, "priors": prior_manifest["sha256"], "data_revision": 1},
    sort_keys=True).encode()).hexdigest()[:16]

def context_sample(history, count, seed, include=None):
    eligible = history[history.item_id.isin(ITEM_TO_ROW)].drop_duplicates('context_key')
    local = np.random.default_rng(seed)
    eligible = eligible.sort_values('context_key').iloc[local.permutation(len(eligible))]
    bounded = eligible.groupby('query_norm', sort=False).head(EXPERIMENT_CONFIG['contexts_per_text'])
    if include is not None:
        bounded = pd.concat([include, bounded], ignore_index=True).drop_duplicates('context_key')
    return bounded.head(count)[[*QUERY_COLS,'query_norm','context_key']].sort_values('context_key').reset_index(drop=True)

# Reserve genuinely new texts before any new fit/selection. Old supervised
# positive IDs are excluded from this audit so their labels cannot be reused.
old_direct_texts = set(training_queries.query_norm) | set(original_training_queries.query_norm)
old_direct_ids = {ITEM_IDS[i] for truth in query_labels(ranker_history, training_queries) for i in truth}
overlap_texts = set(ranker_history.loc[ranker_history.item_id.isin(old_direct_ids), 'query_norm'])
audit_source = ranker_history[~ranker_history.query_norm.isin(old_direct_texts | overlap_texts)]
new_audit = select_query_contexts(audit_source, EXPERIMENT_CONFIG['audit_texts'], SEED + 702)
new_audit_labels = query_labels(ranker_history, new_audit)
clean_history, new_cold_ids = purge_history(ranker_history, new_audit, new_audit_labels, SEED + 703)
expanded_training = context_sample(clean_history, EXPERIMENT_CONFIG['contexts'], SEED + 704, training_queries)
assert not set(expanded_training.query_norm) & set(new_audit.query_norm)
assert not set(clean_history.query_norm) & set(new_audit.query_norm)
assert not new_cold_ids & set(clean_history.item_id)
assert not old_direct_ids & {ITEM_IDS[i] for truth in new_audit_labels for i in truth}
development = pd.concat([validation[[*QUERY_COLS,'query_norm','context_key']], audit_queries, fresh_queries],ignore_index=True)
development_labels = labels + audit_labels + fresh_labels
protocol = {'training_contexts':len(expanded_training),'training_texts':expanded_training.query_norm.nunique(),
    'development_contexts':len(development),'new_audit_texts':len(new_audit),
    'direct_training_text_overlap':0,'direct_training_positive_item_overlap':0,'purged_item_overlap':0}
print('New protocol:',json.dumps(protocol),flush=True)
(CACHE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
expanded_training.to_parquet(CACHE/'training_queries.parquet',index=False)
new_audit.to_parquet(CACHE/'new_audit_queries.parquet',index=False)
semantic_index = SemanticIndex()
all_encoder_queries = pd.concat([expanded_training.query_norm, development.query_norm,
    new_audit.query_norm, queries.query_norm], ignore_index=True)
semantic_index.prepare(all_encoder_queries)
embedding_checksums = {"items": sha256_file(semantic_index.item_path),
    "queries": sha256_file(semantic_index.query_path)}
(CACHE / "embedding_manifest.json").write_text(json.dumps({"model": model_manifest,
    "config": SEMANTIC_CONFIG, "sha256": embedding_checksums}, indent=2), encoding="utf-8")

def prior_scores(records, matrices):
    a=model_scores(prior_legacy,[x[:,:32] for x in matrices])
    b=model_scores(prior_semantic,[x[:,:38] for x in matrices])
    output=[]
    for (_,features),legacy,semantic in zip(records,a,b):
        lex=score_candidates(features,best_config)
        prior=.5/(60+rankdata(-legacy,method='min'))+.5/(60+rankdata(-lex,method='min'))
        output.append(.25/(60+rankdata(-semantic,method='min'))+.75/(60+rankdata(-prior,method='min')))
    return output

def direct_predict(records,scores):
    return [ids[stable_topk(score,50)] for (ids,_),score in zip(records,scores)]

def combined_predict(records,scores,reference,weight):
    if weight==0:return direct_predict(records,reference)
    if weight==1:return direct_predict(records,scores)
    return direct_predict(records,[weight/(60+rankdata(-s,method='min'))+(1-weight)/(60+rankdata(-p,method='min')) for s,p in zip(scores,reference)])

# %% [markdown]
# ## 10. OOF-признаки и трудные отрицательные примеры
#
# Все контексты одного нормализованного текста относятся к одной OOF-группе.
# При построении группы её тексты и 90% positives исключаются из вспомогательной истории.
# Negatives: лексические/семантические лидеры, высокие оценки замороженного бустинга,
# случайный хвост. Пропущенные поиском positives не добавляются в пул принудительно.
# Ранги рассчитываются по полному пулу до sampling. Ненаблюдённые взаимодействия —
# слабые отрицательные примеры; они не являются гарантированно нерелевантными.

# %%
def grouped_training(history,qframe,stage):
    path=CACHE/f'{stage}_{fingerprint}.joblib'
    if USE_CACHE and path.exists():return joblib.load(path)
    truths=query_labels(history,qframe)
    assert all(truths)
    local=np.random.default_rng(SEED+705)
    texts=np.sort(qframe.query_norm.unique())
    assignment={text:int(fold) for text,fold in zip(texts[local.permutation(len(texts))],np.arange(len(texts))%3)}
    xs,ys,weights,groups,audits=[],[],[],[],[]
    for fold in range(3):
        positions=np.flatnonzero(qframe.query_norm.map(assignment).to_numpy()==fold)
        fold_queries=qframe.iloc[positions]
        fold_truths=[truths[i] for i in positions]
        fit,cold=purge_history(history,fold_queries,fold_truths,SEED+706+fold)
        h=HistorySignals(fit)
        print(stage,'fold',fold,'contexts',len(positions),'texts',fold_queries.query_norm.nunique(),flush=True)
        for start in range(0,len(positions),256):
            chunk=fold_queries.iloc[start:start+256]
            records=retrieve_semantic_features(chunk,h,progress_every=1000)
            matrices=[rank_semantic_features(q,r) for q,r in zip(chunk.itertuples(index=False),records)]
            mined=model_scores(prior_legacy,[x[:,:32] for x in matrices])
            for j,((ids,base),features,model_score) in enumerate(zip(records,matrices,mined)):
                truth=fold_truths[start+j]
                target=np.isin(ids,list(truth)); pos=np.flatnonzero(target); neg=np.flatnonzero(~target)
                if not len(pos) or not len(neg):continue
                parts=[neg[stable_topk(score_candidates(base[neg],best_config),EXPERIMENT_CONFIG['hard_source'])],
                    neg[stable_topk(base[neg,7],EXPERIMENT_CONFIG['hard_source'])],
                    neg[stable_topk(model_score[neg],EXPERIMENT_CONFIG['hard_model'])]]
                hard=np.unique(np.concatenate(parts)); tail=np.setdiff1d(neg,hard)
                random=local.choice(tail,min(len(tail),EXPERIMENT_CONFIG['random_tail']),replace=False)
                selected=np.sort(np.concatenate([pos,hard,random])); y=target[selected].astype(np.uint8)
                assert not set(ids[selected[~y.astype(bool)]]) & truth
                xs.append(features[selected]);ys.append(y)
                weights.append(np.where(y,100/max(int(y.sum()),1),100/max(int((1-y).sum()),1)).astype(np.float32))
                groups.append(np.full(len(selected),positions[start+j],dtype=np.int32))
            if start%1024==0:print(stage,'fold',fold,'processed',start+len(chunk),flush=True)
        audits.append({'fold':fold,'contexts':len(positions),'texts':fold_queries.query_norm.nunique(),
            'text_overlap':len(set(fold_queries.query_norm)&set(fit.query_norm)), 'purged_item_overlap':len(cold&set(fit.item_id))})
        del h,fit,records,matrices;gc.collect()
    data={'X':np.concatenate(xs),'y':np.concatenate(ys),'weight':np.concatenate(weights),'group':np.concatenate(groups),'audit':audits}
    data['weight']/=data['weight'].mean()
    save_cache(data,path)
    (CACHE/f'{stage}_leakage_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    print('Training matrix',data['X'].shape,'positives',int(data['y'].sum()),flush=True)
    return data


dataset = grouped_training(clean_history, expanded_training, "expanded_training")

# %% [markdown]
# ## 11. Development: два режима и распределение запросов
#
# Сравниваем новые тексты и знакомые тексты с удержанным контекстом. Во втором режиме
# собственные context labels удалены, 90% целевых ID также удалены из истории; другие
# контексты текста остаются. Это две симуляции тех же 2600 запросов, не 5200 независимых labels.
# Доли режимов берутся только из текстового пересечения benchmark/train. Внутри режимов
# веса согласуют длину запроса, наличие фильтров и локации с benchmark без его разметки.
# Это приближение к распределению теста, не гарантированная оценка закрытой метрики.

# %%
def batched_new_scores(model, features, family, trees=None):
    output=[]
    for start in range(0,len(features),64):
        block=features[start:start+64];x=np.concatenate(block)
        params=({'num_iteration':trees} if family.startswith('lgb') else {'ntree_end':trees}) if trees else {}
        scores=model.predict(x,**params) if family.endswith('rank') else model.predict_proba(x,**params)[:,1]
        output.extend(np.split(scores,np.cumsum([len(a) for a in block])[:-1]))
    return output

def seen_query_history(qframe, truths):
    # Simulate a known text with a held-out context. Own context labels are removed
    # and 90% target IDs are globally purged, but other contexts of the text remain.
    positive_ids=np.array(sorted({ITEM_IDS[i] for truth in truths for i in truth}))
    local=np.random.default_rng(SEED+711)
    cold=set(local.choice(positive_ids,int(.9*len(positive_ids)),replace=False))
    fit=history_all[~history_all.context_key.isin(qframe.context_key)&~history_all.item_id.isin(cold)
        &~history_all.query_norm.isin(new_audit.query_norm)&~history_all.item_id.isin(new_cold_ids)].copy()
    assert not set(qframe.context_key)&set(fit.context_key)
    assert not cold&set(fit.item_id)
    return fit

def strata(frame):
    length=frame.query_norm.str.len().to_numpy()
    return list(zip(np.digitize(length,[15,25]).tolist(),
        (frame.search_infm_params_text.str.len()>0).tolist(),frame.search_location_id.isin(centers.index).tolist()))

def matched_weights(frame, target):
    actual=strata(frame);observed=Counter(actual);desired=Counter(strata(target))
    missing=set(desired)-set(observed)
    assert not missing, ('Missing validation strata',missing)
    weights=np.array([desired.get(k,0)/observed[k]/len(target) for k in actual],dtype=float)
    return weights/weights.sum()

history=HistorySignals(clean_history)
dev_records=cached_hybrid_records(development,history,'new_development')
dev_features=build_rank_feature_records(development,dev_records,'new_development')
reference=prior_scores(dev_records,dev_features)
seen_history=HistorySignals(seen_query_history(development,development_labels))
seen_records=cached_hybrid_records(development,seen_history,'seen_development')
seen_features=build_rank_feature_records(development,seen_records,'seen_development')
seen_reference=prior_scores(seen_records,seen_features)
known=queries.query_norm.isin(history_all.query_norm)
known_fraction=float(known.mean())
unknown_weights=matched_weights(development,queries[~known])
known_weights=matched_weights(development,queries[known])
def selection_metrics(predictions,seen_predictions):
    a=per_query_recall(predictions,development_labels)
    b=per_query_recall(seen_predictions,development_labels)
    return {'development_recall50':float((1-known_fraction)*(a@unknown_weights)+known_fraction*(b@known_weights)),
        'unknown_macro_recall50':float(a.mean()),'known_macro_recall50':float(b.mean())}

ref_metrics=selection_metrics(direct_predict(dev_records,reference),direct_predict(seen_records,seen_reference))
print('V0.3 reference by regime:',ref_metrics,flush=True)
(CACHE/'development_weighting.json').write_text(json.dumps({'known_query_fraction':known_fraction,
    'validation_contexts_per_regime':len(development),'strata':['query length','filters present','location in corpus'],
    'unknown_effective_sample_size':float(1/(unknown_weights**2).sum()),
    'known_effective_sample_size':float(1/(known_weights**2).sum()),'reference':ref_metrics},indent=2),encoding='utf-8')

# %% [markdown]
# ## 12. Сравнение алгоритмов и объёма обучения
#
# LightGBM LambdaRank и бинарный classifier; CatBoost YetiRank/NDCG и Logloss.
# Каждый вариант проверен на 4000 и расширенных контекстах, при 200/400/800 деревьях.
# RRF с reference: веса 25/50/75/100%. Reference также участвует как fallback.
# Выбор только по development. Дополнительное условие: падение обычного macro Recall
# новых текстов не превышает 0.5 п.п. При выборе новые 800 запросов не использовались.
#
# По умолчанию Run All обучает уже выбранный LambdaRank: 200 деревьев, вес 75%.
# Полный сравнительный эксперимент сохранён в development_experiments_v2.csv;
# его можно повторить с AVITO_RUN_MODEL_SEARCH=1 до запуска ядра. Это не требуется
# для воспроизведения финального ответа. Фиксированы гиперпараметры, а не ответы.

# %%
if RUN_MODEL_SEARCH:
    base_groups=set(expanded_training.index[expanded_training.context_key.isin(training_queries.context_key)])
    subset=np.isin(dataset['group'],list(base_groups))
    experiments=[{'model':'prior_v3','family':'prior','scale':'4k','trees':120,'weight':0.,**ref_metrics}]
    model_paths={}
    for scale,mask in [('4k',subset),('expanded',np.ones(len(dataset['y']),dtype=bool))]:
        x,y,w,g=dataset['X'][mask],dataset['y'][mask],dataset['weight'][mask],dataset['group'][mask]
        order=np.argsort(g,kind='stable');x,y,w,g=x[order],y[order],w[order],g[order]
        w /= w.mean()
        _,sizes=np.unique(g,return_counts=True)
        for family in ['lgb_rank','lgb_class','cat_rank','cat_class']:
            name=f'{family}_{scale}';path=CACHE/f'{name}_800_{fingerprint}.joblib';started=time.perf_counter()
            if USE_CACHE and path.exists():model=joblib.load(path)
            else:
                if family.startswith('lgb'):
                    cls=lgb.LGBMRanker if family=='lgb_rank' else lgb.LGBMClassifier
                    model=cls(n_estimators=800,num_leaves=31,learning_rate=.05,max_bin=127,min_child_samples=50,
                        reg_lambda=10,random_state=SEED,n_jobs=8,verbosity=-1,deterministic=True,force_col_wise=True,
                        **({'lambdarank_truncation_level':55,'label_gain':[0,1]} if family=='lgb_rank' else {}))
                    model.fit(x,y,**({'group':sizes} if family=='lgb_rank' else {'sample_weight':w}))
                else:
                    cls=CatBoostRanker if family=='cat_rank' else CatBoostClassifier
                    model=cls(iterations=800,depth=6,learning_rate=.05,l2_leaf_reg=10,random_seed=SEED,
                        thread_count=8,allow_writing_files=False,verbose=100,
                        loss_function='YetiRank:mode=NDCG;top=50' if family=='cat_rank' else 'Logloss')
                    model.fit(x,y,**({'group_id':g} if family=='cat_rank' else {'sample_weight':w}))
                save_cache(model,path)
            model_paths[name]=str(path)
            for trees in [200,400,800]:
                scores=batched_new_scores(model,dev_features,family,trees)
                seen_scores=batched_new_scores(model,seen_features,family,trees)
                for weight in [.25,.5,.75,1.]:
                    metrics=selection_metrics(combined_predict(dev_records,scores,reference,weight),
                        combined_predict(seen_records,seen_scores,seen_reference,weight))
                    experiments.append({'model':name,'family':family,'scale':scale,'trees':trees,'weight':weight,**metrics})
            pd.DataFrame(experiments).sort_values('development_recall50',ascending=False).to_csv(CACHE/'development_experiments_v2.csv',index=False)
            print(name,'best matched dev',max(r['development_recall50'] for r in experiments if r['model']==name),
                'seconds',round(time.perf_counter()-started,1),flush=True)
        del x,y,w,g;gc.collect()
    # Require the unknown-query stress check to remain within 0.5pp of the reference.
    eligible=[r for r in experiments if r['unknown_macro_recall50']>=ref_metrics['unknown_macro_recall50']-.005]
    best=sorted(eligible,key=lambda r:(-r['development_recall50'],r['model'],r['trees'],r['weight']))[0]
    (CACHE/'choice_v2.json').write_text(json.dumps({'selected':best,'model_paths':model_paths,'protocol':protocol,
        'config':EXPERIMENT_CONFIG,'known_query_fraction':known_fraction},indent=2),encoding='utf-8')
    print('Frozen before reserved audit:',best,flush=True)

else:
    # Hyperparameters frozen from the development comparison, before the audit.
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

# %% [markdown]
# ## 13. Зарезервированные 800 запросов: оценка один раз
#
# Конфигурация уже зафиксирована. Метрика считается среди всего benchmark-корпуса,
# включая positives, пропущенные candidate generation. Bootstrap разницы парный,
# по запросам; audit не участвует в выборе параметров.

# %%
test_records=cached_hybrid_records(new_audit,history,'new_audit')
test_features=build_rank_feature_records(new_audit,test_records,'new_audit')
test_reference=prior_scores(test_records,test_features)
if best['family']=='prior':predictions=direct_predict(test_records,test_reference)
else:
    model=joblib.load(model_paths[best['model']])
    scores=batched_new_scores(model,test_features,best['family'],best['trees'])
    predictions=combined_predict(test_records,scores,test_reference,best['weight'])
a=per_query_recall(direct_predict(test_records,test_reference),new_audit_labels)
b=per_query_recall(predictions,new_audit_labels)
local=np.random.default_rng(SEED+707);delta=b-a
boot=[float(local.choice(delta,len(delta),replace=True).mean()) for _ in range(4000)]
result={'queries':len(new_audit),'reference_recall50':float(a.mean()),'selected_recall50':float(b.mean()),
    'delta':float(delta.mean()),'paired_delta_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
    'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),'used_for_selection':False,
    'pool_recall':float(per_query_recall([r[0] for r in test_records],new_audit_labels).mean()),'choice':best}
(CACHE/'audit_metrics_v2.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
pd.DataFrame({'query':new_audit.search_query,'reference':a,'selected':b}).to_csv(CACHE/'audit_comparison_v2.csv',index=False)
print('RESERVED AUDIT',json.dumps(result),flush=True)

# %%
# Separate retrieval losses from losses while reducing the pool to 50.
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

# %% [markdown]
# ## 14. Финальное переобучение и answer.csv
#
# После всех оценок отложенные части возвращаются в финальное OOF-обучение.
# Параметры зафиксированы; финальная модель не оценивается на уже использованных labels.
# Для benchmark prior заменяется финальным prior v0.3.0. Новая модель смешивается
# с ним с выбранным весом. Ответ проверяется по ID исходных файлов и повторно читается.

# %%
def validate_answer(answer_frame, expected_query_ids, allowed_item_ids):
    if answer_frame.columns.tolist() != ["query_id", "answer"]:
        raise ValueError("Expected exactly query_id,answer columns")
    if answer_frame.isna().any().any():
        raise ValueError("Missing values in answer")
    expected = set(expected_query_ids)
    if len(answer_frame) != len(expected) or not answer_frame.query_id.is_unique:
        raise ValueError("Missing or duplicated query rows")
    if set(answer_frame.query_id) != expected:
        raise ValueError("Query IDs do not match benchmark")
    if not answer_frame.query_id.str.len().eq(16).all():
        raise ValueError("Invalid query_id length")
    allowed = set(allowed_item_ids)
    for query_id, text in answer_frame.itertuples(index=False, name=None):
        ids = text.split(" ") if text else []
        if len(ids) > 50 or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid answer count/duplicates for {query_id}")
        if any(re.fullmatch(r"[0-9a-f]{16}", item_id) is None for item_id in ids):
            raise ValueError(f"Invalid item_id format for {query_id}")
        if not set(ids).issubset(allowed):
            raise ValueError(f"Unknown item IDs for {query_id}")
    return True
# Final fitting uses the selected architecture and never tunes on audit results.
final_queries = pd.concat([expanded_training if best['scale']=='expanded' else training_queries,
    development, new_audit], ignore_index=True)[[*QUERY_COLS,'query_norm','context_key']]
final_queries = final_queries.drop_duplicates('context_key').sort_values('context_key').reset_index(drop=True)
final_model_files = []
final_model = None
if best['family'] != 'prior':
    final_path = CACHE / f"final_ranker_{fingerprint}.joblib"
    if USE_CACHE and final_path.exists():
        final_model = joblib.load(final_path)
    else:
        # Release evaluation matrices before the larger final training pass.
        del dataset, dev_features, seen_features, test_features
        gc.collect()
        final_data = grouped_training(history_all, final_queries, 'final_training')
        x,y,w,g = (final_data[key] for key in ['X','y','weight','group'])
        order = np.argsort(g,kind='stable')
        x,y,w,g = x[order],y[order],w[order],g[order]
        _,sizes = np.unique(g,return_counts=True)
        family = best['family']
        if family.startswith('lgb'):
            cls = lgb.LGBMRanker if family=='lgb_rank' else lgb.LGBMClassifier
            final_model = cls(n_estimators=best['trees'],num_leaves=31,learning_rate=.05,max_bin=127,
                min_child_samples=50,reg_lambda=10,random_state=SEED,n_jobs=8,
                verbosity=-1,deterministic=True,force_col_wise=True,
                **({'lambdarank_truncation_level':55,'label_gain':[0,1]} if family=='lgb_rank' else {}))
            final_model.fit(x,y,**({'group':sizes} if family=='lgb_rank' else {'sample_weight':w}))
        else:
            cls = CatBoostRanker if family=='cat_rank' else CatBoostClassifier
            final_model = cls(iterations=best['trees'],depth=6,learning_rate=.05,l2_leaf_reg=10,
                random_seed=SEED,thread_count=8,allow_writing_files=False,verbose=100,
                loss_function='YetiRank:mode=NDCG;top=50' if family=='cat_rank' else 'Logloss')
            final_model.fit(x,y,**({'group_id':g} if family=='cat_rank' else {'sample_weight':w}))
        save_cache(final_model,final_path)
        del final_data,x,y,w,g
        gc.collect()
    final_model_files.append(final_path.name)
prior_legacy = joblib.load(PRIORS_DIR/'final_legacy.joblib')
prior_semantic = joblib.load(PRIORS_DIR/'final_semantic.joblib')
benchmark_history = HistorySignals(history_all)
benchmark_records = cached_hybrid_records(queries,benchmark_history,'benchmark')
benchmark_features = build_rank_feature_records(queries,benchmark_records,'benchmark')
benchmark_reference = prior_scores(benchmark_records,benchmark_features)
if final_model is None:
    predictions = direct_predict(benchmark_records,benchmark_reference)
else:
    benchmark_scores = batched_new_scores(final_model,benchmark_features,best['family'])
    predictions = combined_predict(benchmark_records,benchmark_scores,benchmark_reference,best['weight'])
answer = pd.DataFrame({'query_id':queries.query_id.astype(str),
    'answer':[' '.join(ITEM_IDS[indices]) for indices in predictions]})
validate_answer(answer,queries.query_id,ITEM_IDS)
assert answer.answer.str.split().str.len().eq(50).all()
answer.to_csv(ROOT/'answer.csv',index=False,encoding='utf-8',lineterminator='\n')
reloaded = pd.read_csv(ROOT/'answer.csv',dtype=str,keep_default_na=False)
assert reloaded.equals(answer)
validate_answer(reloaded,queries.query_id,ITEM_IDS)
repeat = direct_predict(benchmark_records,benchmark_reference) if final_model is None else combined_predict(
    benchmark_records,batched_new_scores(final_model,benchmark_features,best['family']),benchmark_reference,best['weight'])
assert all(np.array_equal(a,b) for a,b in zip(predictions,repeat))
metrics = {'development':best,'reference_development':ref_metrics,'new_audit':result,
    'protocol':protocol,'known_query_fraction':known_fraction,'final_contexts':len(final_queries),
    'final_texts':int(final_queries.query_norm.nunique())}
(CACHE/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
manifest = {'config':CONFIG,'experiment_config':EXPERIMENT_CONFIG,'ranker_choice':best,
    'semantic_config':SEMANTIC_CONFIG,'prior_manifest':prior_manifest,'frozen_priors':True,
    'model_manifest':model_manifest,'embedding_sha256':embedding_checksums,
    'input_sha256':input_hashes,'fingerprint':fingerprint,'code_sha256':code_hash,
    'final_model_files':final_model_files,'answer_sha256':sha256_file(ROOT/'answer.csv'),'metrics':metrics,
    'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,
        'sklearn':sklearn.__version__,'torch':torch.__version__,'transformers':transformers.__version__,
        'lightgbm':lgb.__version__,'catboost':__import__('catboost').__version__,'device':DEVICE}}
(CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Saved',ROOT/'answer.csv','; queries',len(answer),'; SHA256',manifest['answer_sha256'])

# %% [markdown]
# ## 15. Интерпретация и воспроизводимость
#
# Отправлять answer.csv. Код, описание, локальные веса/векторы и dependencies приложены.
# Прежний реальный результат платформы v0.3.0 — Recall@50 0.858569. Новая версия
# выбирается по локальным development-данным; её платформенная метрика пока неизвестна.
# Режим known/unknown и согласование распределений — проверяемые гипотезы.
# Разметка отражает пользовательский выбор, ограничена корпусным пересечением и неполна.
#
# Использованы NumPy, pandas, SciPy, PyArrow, scikit-learn, Snowball, joblib,
# PyTorch/Transformers, CatBoost и LightGBM. Модель E5 и алгоритмы открытые;
# чужие решения задания не использовались. Источники:
# - [Multilingual E5](https://huggingface.co/intfloat/multilingual-e5-small)
# - [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking)
# - [LightGBM ranking API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)
# - [BM25](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)
