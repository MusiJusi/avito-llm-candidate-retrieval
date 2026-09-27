# %% [markdown]
# # Кандидатогенерация объявлений услуг Авито
#
# **Цель:** вернуть 50 уникальных `item_id` для каждого запроса; метрика — macro Recall@50.
# Версия с обучаемым отбором: прежние BM25/TF-IDF источники создают кандидатов,
# градиентный бустинг выбирает 50 с учётом текстовых и структурных признаков.
# Все вычисления локальные. Внешние API и разметка benchmark не используются.
#
# **Запуск:** положить три исходных Parquet рядом с ноутбуком, установить
# `requirements.txt` в окружение его ядра и выполнить **Restart Kernel → Run All**.
# На текущем компьютере зависимости уже находятся в `.inspection_deps`.
# Ноутбук сам строит индексы и сохраняет `answer.csv`, отчёт и анализ ошибок.
# Кеш ускоряет повторный запуск; его можно удалить и пересчитать всё из исходных файлов.
#
# Обучение и поиск выполняются на CPU. Семантического поиска в этой версии нет.
# `search_category` не используется в scoring; алгоритм создания пула не изменён.

# %%
from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "train.parquet").exists() and (ROOT / "AvitoTest/train.parquet").exists():
    ROOT = ROOT / "AvitoTest"
if (ROOT / ".inspection_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".inspection_deps"))

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
    "version": "learned-selection-v1",
}
RANKER_CONFIG = {
    "training_queries": 4000, "audit_queries": 600, "folds": 3,
    "hard_negatives": 100, "random_negatives": 60,
    "iterations": [120, 240], "leaf_options": [15, 31],
    "learning_rate": 0.08, "l2_regularization": 10.0,
    "blend_options": [0.5, 0.75, 1.0], "feature_version": 1,
}
CACHE = ROOT / "artifacts" / CONFIG["version"]
CACHE.mkdir(parents=True, exist_ok=True)
USE_CACHE = os.environ.get("AVITO_REBUILD_CACHE", "0") != "1"
threadpool_limits(limits=8)
np.random.seed(SEED)
print("Project:", ROOT)
print("Versions:", {"python": sys.version.split()[0], "pandas": pd.__version__,
                    "numpy": np.__version__, "sklearn": sklearn.__version__})

# %% [markdown]
# ## 1. Данные и проверка схемы
#
# Идентификаторы сохраняются строками. Цена и координаты в Parquet представлены
# Decimal: явно переводим их в числа, чтобы не хранить множество Python-объектов.
# У объявлений сортируем строки по `item_id`: этот порядок также разрешает ничьи
# при отборе кандидатов. Тексты отсутствующих описаний заменяем пустой строкой.

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
# ## 2. Нормализация и первичный анализ
#
# Для сопоставления текстов нормализуем Unicode, регистр, `ё/е` и пробелы.
# ID, географические признаки и числа категорий не нормализуем как текст.
# Поиск дополнительно использует русский Snowball stemmer и символьные n-граммы.

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
# ## 3. Валидация: новые тексты запросов и преимущественно новые объявления
#
# Из train берём запросы, имеющие положительные объявления в benchmark-корпусе.
# Выбираем случайные **уникальные нормализованные тексты**, затем один контекст
# на текст и все его известные положительные объявления, присутствующие в корпусе.
# Все взаимодействия с выбранными текстами исключаем из обучающей истории.
# Дополнительно удаляем из истории взаимодействия с 90% выбранных положительных
# item_id, чтобы проверить обобщение на новые объявления, а не запоминание ID.
#
# 700 запросов — development для выбора весов; 700 — отложенный holdout для одной
# итоговой оценки. Обе части исключены из обучающей истории с самого начала.
# Тексты и метаданные **всего корпуса** доступны для построения поискового индекса:
# это соответствует задаче. В holdout не подмешиваем релевантные объявления в выдачу.
#
# **Ограничения:** это стресс-проверка новых текстов, а benchmark содержит и знакомые.
# Разметка здесь ограничена пересечением с корпусом; это другой срез исходного train.
# Метрика не является гарантией результата платформы. IDF по доступному корпусу
# не использует пользовательские выборы и не является утечкой меток.

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
# Два BM25-индекса: отдельно заголовок и документ из заголовка, параметров и описания.
# Ограничиваем длинные поля, чтобы прайс-листы и рекламные блоки не определяли всю
# длину документа. Символьный TF-IDF заголовков дополняет BM25 при опечатках.
#
# Реализуем BM25 через разреженные матрицы SciPy: храним веса только ненулевых
# термов и не перебираем Python-циклом все объявления на каждый запрос.
# Параметры: k1=1.2, b=0.75, положительный IDF log(1 + (N-df+0.5)/(df+0.5)).
# Для TF-IDF используем scikit-learn. Слова приводим к основам русским Snowball.
#
# Кеш связан с содержимым исходных файлов и конфигурацией. Он содержит только
# производные индексы; загрузка joblib предназначена для своих локальных артефактов.

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
                                        "ranker": RANKER_CONFIG,
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
# ## 5. География и перенос подкатегории из обучающей истории
#
# Совпадение локации — сильный сигнал, но 17% положительных пар имеют разные ID.
# Поэтому географию учитываем мягко: точное совпадение, расстояние до медианного
# центра локации и наблюдавшиеся переходы search_location → item_location.
# Центры локаций вычисляем из доступных координат корпуса, переходы — только из fit.
# Если географической информации нет, не штрафуем объявления произвольно.
#
# Дополнительный сигнал: находим похожие тексты запросов в fit по символьному TF-IDF
# и усредняем распределения их выбранных подкатегорий. Это простой обучаемый
# классификатор намерения. Он работает и для объявлений, которых не было в train.
# Никаких ручных списков ответов или специальных правил по benchmark query_id нет.

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

started = time.perf_counter()
history_model = HistorySignals(history_fit)
print("History signals prepared in", round(time.perf_counter() - started, 1), "seconds")

# %% [markdown]
# ## 6. Кандидаты и подбор весов на development
#
# Сначала объединяем top-500 каждого текстового источника и варианты с географией.
# Затем на этом фиксированном пуле сравниваем веса. Для каждого запроса оценки
# источников нормируем на их максимум; география и подкатегория применяются как
# мягкие множители. Объявления вне города/предсказанной подкатегории не запрещены.
# Параметры поискового фильтра дают небольшой дополнительный текстовый сигнал.
#
# Помимо Recall@50 измеряем полноту объединённого пула: это верхняя граница любого
# последующего отбора из него. Положительные метки не используются при создании пула.

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

validation_cache = CACHE / f"validation_features_{fingerprint}.joblib"
if USE_CACHE and validation_cache.exists():
    validation_features = joblib.load(validation_cache)
    print("Loaded validation features")
else:
    validation_features = retrieve_features(validation, history_model)
    save_cache(validation_features, validation_cache)

experiments = [
    {"name": "title_bm25", "title": 1, "body": 0, "char": 0, "filter": 0, "geo": 0, "micro": 0},
    {"name": "body_bm25", "title": 0, "body": 1, "char": 0, "filter": 0, "geo": 0, "micro": 0},
    {"name": "char_tfidf", "title": 0, "body": 0, "char": 1, "filter": 0, "geo": 0, "micro": 0},
]
for text_weights in [(0.5, 0.3, 0.2), (0.3, 0.5, 0.2), (0.65, 0.2, 0.15)]:
    for geo_weight in [0, 0.5, 1.0, 2.0]:
        for micro_weight in [0, 0.5, 1.0]:
            experiments.append({"name": f"hybrid_{len(experiments)}", "title": text_weights[0],
                                "body": text_weights[1], "char": text_weights[2],
                                "filter": 0.03, "geo": geo_weight, "micro": micro_weight})
dev_records = [validation_features[i] for i in dev_indices]
dev_labels = [labels[i] for i in dev_indices]
results = []
for config in experiments:
    recall = per_query_recall(predict_from_features(dev_records, config), dev_labels)
    results.append({**config, "development_recall50": float(recall.mean())})
results = pd.DataFrame(results).sort_values(["development_recall50", "name"], ascending=[False, True])
best_config = {key: results.iloc[0][key] for key in experiments[0]}
results.to_csv(CACHE / "development_experiments.csv", index=False)
(CACHE / "best_config.json").write_text(json.dumps(best_config, indent=2), encoding="utf-8")
print(results.head(10).to_string(index=False))
print("\nBaselines:")
print(results[results.name.isin(["title_bm25", "body_bm25", "char_tfidf"])].to_string(index=False))
print("Development pool recall:", per_query_recall([r[0] for r in dev_records], dev_labels).mean())

# %% [markdown]
# ## 7. Независимые обучающие запросы и новая контрольная выборка
#
# Старые 700 holdout-запросов уже анализировались при создании baseline. Сохраняем
# их для парного сравнения, но дополнительно резервируем 600 новых audit-запросов.
# Их тексты и 90% положительных item_id исключаем из истории до подготовки ranker.
# Audit не используется для выбора моделей, весов или количества деревьев.
#
# Из оставшейся истории выбираем до 4 000 текстов, по одному контексту на текст.
# Три OOF-группы разделены по нормализованному тексту. При создании признаков группы
# её тексты и 90% её положительных item_id удалены из вспомогательной истории.
# Это имитирует новые запросы/объявления и исключает запоминание собственных выборов.
# Пропущенные retrieval-пулом положительные объявления **не добавляем** в пул.

# %%
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
training_queries = select_query_contexts(ranker_history, RANKER_CONFIG["training_queries"], SEED + 103)
assert not set(training_queries.query_norm) & set(validation.query_norm)
assert not set(training_queries.query_norm) & set(audit_queries.query_norm)
assert not set(ranker_history.query_norm) & set(validation.query_norm)
assert all(audit_labels)
training_queries.to_parquet(CACHE / "ranker_training_queries.parquet", index=False)
audit_queries.to_parquet(CACHE / "audit_queries.parquet", index=False)
print("Ranker training queries:", len(training_queries), "; fresh audit queries:", len(audit_queries))
print("Training/validation/audit text intersections: 0")

# %% [markdown]
# ## 8. Признаки обучаемого отбора
#
# Бустинг получает оценки прежних поисков, географическую совместимость,
# покрытие слов запроса в заголовке/документе, ранг baseline, расстояние, рейтинг,
# отзывы, цену, флаги связи и длины текстов. ID запросов, объявлений, городов и
# категорий не передаются модели. `search_category` не добавляется ни как признак,
# ни как фильтр. Числовые ID не трактуются как порядковые величины.
#
# Метки используются только после получения кандидатов и вычисления признаков.
# Пропуски числовых признаков сохраняются NaN: HistGradientBoosting умеет их обрабатывать.
# Строка «рейтинг ... 4» в фильтре даёт числовой порог и признак его выполнения,
# без жёсткого отбрасывания объявлений.

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
    assert result.shape == (n, len(RANK_FEATURE_NAMES))
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

def build_oof_training(history, query_frame, stage):
    path = CACHE / f"{stage}_training_{fingerprint}.joblib"
    if USE_CACHE and path.exists():
        return joblib.load(path)
    truths = query_labels(history, query_frame)
    assert all(truths)
    local_rng = np.random.default_rng(SEED + 200)
    assignments = np.empty(len(query_frame), dtype=int)
    assignments[local_rng.permutation(len(query_frame))] = np.arange(len(query_frame)) % RANKER_CONFIG["folds"]
    xs, ys, weights, groups, audit = [], [], [], [], []
    for fold in range(RANKER_CONFIG["folds"]):
        positions = np.flatnonzero(assignments == fold)
        fold_queries = query_frame.iloc[positions]
        fold_truths = [truths[i] for i in positions]
        clean_history, cold_ids = purge_history(history, fold_queries, fold_truths, SEED + 300 + fold)
        print(f"{stage}: fold {fold + 1}/{RANKER_CONFIG['folds']}; queries={len(positions)}; history={len(clean_history)}", flush=True)
        fold_history_model = HistorySignals(clean_history)
        records = retrieve_features(fold_queries, fold_history_model, progress_every=500)
        missing = 0
        for j, (query, record, truth) in enumerate(zip(fold_queries.itertuples(index=False), records, fold_truths)):
            candidates, base = record
            selected, targets = hard_negative_sample(candidates, base, truth, local_rng)
            if not len(selected):
                missing += 1
                continue
            # Ranks are computed against the full pool before negative sampling.
            features = rank_features(query, record)[selected]
            y = targets[selected].astype(np.uint8)
            n_positive, n_negative = int(y.sum()), int((1 - y).sum())
            # Every query has the same total loss weight; its positives share weight.
            w = np.where(y == 1, 100.0 / n_positive, 100.0 / n_negative).astype(np.float32)
            xs.append(features)
            ys.append(y)
            weights.append(w)
            groups.append(np.full(len(selected), positions[j], dtype=np.int32))
        audit.append({"fold": fold, "queries": len(positions), "history_pairs": len(clean_history),
                      "purged_positive_item_ids": len(cold_ids), "query_text_overlap": 0,
                      "purged_item_overlap": 0, "queries_without_retrieved_positive": missing,
                      "pool_recall": float(per_query_recall([r[0] for r in records], fold_truths).mean())})
        del fold_history_model, clean_history, records
        gc.collect()
    dataset = {"X": np.concatenate(xs), "y": np.concatenate(ys), "weight": np.concatenate(weights),
               "group": np.concatenate(groups), "audit": audit}
    dataset["weight"] /= dataset["weight"].mean()
    assert len(np.unique(dataset["group"])) == sum(x["queries"] - x["queries_without_retrieved_positive"] for x in audit)
    save_cache(dataset, path)
    (CACHE / f"{stage}_leakage_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(stage, "training matrix:", dataset["X"].shape, "; positives:", int(dataset["y"].sum()))
    return dataset

training_data = build_oof_training(ranker_history, training_queries, "evaluation")
print("Features:", len(RANK_FEATURE_NAMES), "; training examples:", len(training_data["y"]))

# %% [markdown]
# ## 9. Выбор модели только по development
#
# Используем HistGradientBoostingClassifier с взвешенной бинарной log-loss.
# Это pointwise-обучение отбора, а не прямая оптимизация Recall@50. Неизвестные
# взаимодействия считаем слабыми отрицательными примерами, а не доказанно нерелевантными.
# Каждый запрос имеет одинаковый суммарный вес; все его известные найденные positives
# делят положительную часть веса. Случайный row-wise early stopping выключен.
#
# Проверяем 15/31 лист и 120/240 деревьев. Дополнительно сравниваем смешивание рангов
# модели и baseline (RRF, константа 60). Выбор — только по macro Recall@50 development.
# Пул и знаменатель Recall неизменны. Если обучение не улучшит development,
# экспорт автоматически сохранит baseline. Старый holdout и новый audit здесь не читаются.

# %%
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

dev_rank_features = build_rank_feature_records(validation.iloc[dev_indices], dev_records, "development")
ranker_experiments = [{"model": "baseline", "leaves": 0, "iterations": 0, "alpha": 0.0,
                       "development_recall50": float(results.iloc[0].development_recall50)}]
model_paths = {}
for leaves in RANKER_CONFIG["leaf_options"]:
    model = HistGradientBoostingClassifier(learning_rate=RANKER_CONFIG["learning_rate"],
        max_leaf_nodes=leaves, min_samples_leaf=30, l2_regularization=RANKER_CONFIG["l2_regularization"],
        max_bins=127, early_stopping=False, warm_start=True, random_state=SEED)
    for iterations in RANKER_CONFIG["iterations"]:
        name = f"hgb_leaves{leaves}_trees{iterations}"
        path = CACHE / f"{name}_{fingerprint}.joblib"
        started = time.perf_counter()
        if USE_CACHE and path.exists():
            model = joblib.load(path)
        else:
            model.set_params(max_iter=iterations)
            model.fit(training_data["X"], training_data["y"], sample_weight=training_data["weight"])
            save_cache(model, path)
        model_paths[name] = path
        scores = model_scores(model, dev_rank_features)
        for alpha in RANKER_CONFIG["blend_options"]:
            predictions = blend_predictions(dev_records, scores, alpha)
            value = float(per_query_recall(predictions, dev_labels).mean())
            ranker_experiments.append({"model": name, "leaves": leaves, "iterations": iterations,
                                       "alpha": alpha, "development_recall50": value})
        print(name, "seconds:", round(time.perf_counter() - started, 1),
              "best development recall:", max(x["development_recall50"] for x in ranker_experiments if x["model"] == name), flush=True)
ranker_results = pd.DataFrame(ranker_experiments).sort_values(
    ["development_recall50", "iterations", "leaves", "alpha"], ascending=[False, True, True, True])
ranker_choice = {"model": str(ranker_results.iloc[0].model), "leaves": int(ranker_results.iloc[0].leaves),
                 "iterations": int(ranker_results.iloc[0].iterations), "alpha": float(ranker_results.iloc[0].alpha),
                 "development_recall50": float(ranker_results.iloc[0].development_recall50)}
evaluation_ranker = None if ranker_choice["model"] == "baseline" else joblib.load(model_paths[ranker_choice["model"]])
ranker_results.to_csv(CACHE / "ranker_development_experiments.csv", index=False)
(CACHE / "ranker_choice.json").write_text(json.dumps(ranker_choice, indent=2), encoding="utf-8")
(CACHE / "feature_names.json").write_text(json.dumps(RANK_FEATURE_NAMES, indent=2), encoding="utf-8")
print(ranker_results.to_string(index=False))

def selected_predictions(query_frame, records, model, stage):
    if ranker_choice["alpha"] == 0:
        return predict_from_features(records, best_config)
    enriched = build_rank_feature_records(query_frame, records, stage)
    scores = model_scores(model, enriched)
    return blend_predictions(records, scores, ranker_choice["alpha"])

# %% [markdown]
# ## 10. Отложенная оценка и анализ ошибок
#
# Конфигурация уже выбрана на development. Holdout используется для итоговой оценки,
# а не перебора весов. Доверительный интервал получаем bootstrap по запросам.
# В таблице ошибок различаем потерю при создании пула и при отборе 50 кандидатов.
# Для ручного анализа сохраняем текст запроса и заголовки пропущенных объявлений.

# %%
holdout_records = [validation_features[i] for i in holdout_indices]
holdout_labels = [labels[i] for i in holdout_indices]
holdout_baseline_predictions = predict_from_features(holdout_records, best_config)
holdout_predictions = selected_predictions(validation.iloc[holdout_indices], holdout_records, evaluation_ranker, "holdout")
holdout_baseline_recall = per_query_recall(holdout_baseline_predictions, holdout_labels)
holdout_recall = per_query_recall(holdout_predictions, holdout_labels)
pool_recall = per_query_recall([r[0] for r in holdout_records], holdout_labels)
boot_rng = np.random.default_rng(SEED)
bootstrap = np.mean(boot_rng.choice(holdout_recall, size=(2000, len(holdout_recall)), replace=True), axis=1)
metrics = {
    "protocol": "cold_query_text_90pct_purged_positive_items_full_benchmark_corpus",
    "development_queries": len(dev_indices), "holdout_queries": len(holdout_indices),
    "development_recall50": ranker_choice["development_recall50"],
    "baseline_development_recall50": float(results.iloc[0].development_recall50),
    "baseline_holdout_recall50": float(holdout_baseline_recall.mean()),
    "holdout_recall50": float(holdout_recall.mean()),
    "holdout_recall50_bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
    "holdout_pool_recall": float(pool_recall.mean()),
    "mean_pool_size": float(np.mean([len(r[0]) for r in holdout_records])),
    "best_config": best_config, "ranker_choice": ranker_choice,
}
print(json.dumps(metrics, indent=2))
error_rows = []
for local_i, index in enumerate(holdout_indices):
    query = validation.iloc[index]
    truth = labels[index]
    predicted = set(map(int, holdout_predictions[local_i]))
    pool = set(map(int, holdout_records[local_i][0]))
    missed = sorted(truth - predicted)
    error_rows.append({
        "search_query": query.search_query, "search_location_id": int(query.search_location_id),
        "search_infm_params_text": query.search_infm_params_text,
        "recall50": float(holdout_recall[local_i]), "pool_recall": float(pool_recall[local_i]),
        "baseline_recall50": float(holdout_baseline_recall[local_i]),
        "positives": len(truth), "lost_before_pool": len(truth - pool),
        "lost_at_selection": len((truth & pool) - predicted),
        "positive_same_location": all(ITEM_LOCS[j] == query.search_location_id for j in truth),
        "missed_item_ids": " ".join(ITEM_IDS[missed]),
        "missed_titles": " | ".join(items.item_title_raw.iloc[missed].astype(str)),
        "top5_titles": " | ".join(items.item_title_raw.iloc[holdout_predictions[local_i][:5]].astype(str)),
    })
error_analysis = pd.DataFrame(error_rows)
error_analysis.to_csv(CACHE / "holdout_error_analysis.csv", index=False, encoding="utf-8")
print("\nRecall by location match:")
print(error_analysis.groupby("positive_same_location").recall50.agg(["count", "mean"]).to_string())
print("\nExamples to inspect:")
print(error_analysis[error_analysis.recall50 < 1][["search_query", "recall50", "pool_recall", "missed_titles"]].head(12).to_string(index=False))
(CACHE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# %% [markdown]
# ## 11. Дополнительная проверка на новых 600 audit-запросах
#
# Модель и смешивание уже зафиксированы. Этот срез не используется для настройки.
# Baseline и ranker получают один и тот же пул, построенный без audit-взаимодействий.
# Bootstrap разницы считается парно по запросам, а не по строкам кандидатов.

# %%
audit_history_model = HistorySignals(ranker_history)
audit_cache = CACHE / f"audit_features_{fingerprint}.joblib"
if USE_CACHE and audit_cache.exists():
    audit_records = joblib.load(audit_cache)
else:
    audit_records = retrieve_features(audit_queries, audit_history_model)
    save_cache(audit_records, audit_cache)
audit_baseline = predict_from_features(audit_records, best_config)
audit_predictions = selected_predictions(audit_queries, audit_records, evaluation_ranker, "audit")
audit_base_recall = per_query_recall(audit_baseline, audit_labels)
audit_model_recall = per_query_recall(audit_predictions, audit_labels)
paired_difference = audit_model_recall - audit_base_recall
audit_rng = np.random.default_rng(SEED + 400)
bootstrap_difference = audit_rng.choice(paired_difference, size=(3000, len(paired_difference)), replace=True).mean(axis=1)
audit_metrics = {
    "queries": len(audit_queries), "baseline_recall50": float(audit_base_recall.mean()),
    "selected_recall50": float(audit_model_recall.mean()),
    "delta": float(paired_difference.mean()), "paired_delta_bootstrap95": np.quantile(bootstrap_difference, [.025, .975]).tolist(),
    "improved_queries": int((paired_difference > 0).sum()), "worsened_queries": int((paired_difference < 0).sum()),
    "pool_recall": float(per_query_recall([r[0] for r in audit_records], audit_labels).mean()),
    "used_for_model_selection": False,
}
metrics["fresh_audit"] = audit_metrics
print(json.dumps(audit_metrics, indent=2))
audit_report = audit_queries[["search_query", "search_location_id", "context_key"]].copy()
audit_report["baseline_recall50"] = audit_base_recall
audit_report["selected_recall50"] = audit_model_recall
audit_report["delta"] = paired_difference
audit_report.to_csv(CACHE / "audit_query_comparison.csv", index=False)
(CACHE / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
del training_data, audit_history_model
gc.collect()

# %% [markdown]
# ## 12. Финальное переобучение и ответ для benchmark
#
# После выбора параметров возвращаем всю обучающую историю, включая локальную
# валидацию: это допустимо, поскольку benchmark-разметка неизвестна и не используется.
# Текстовые индексы корпуса остаются теми же. Возвращаем ровно 50 кандидатов:
# при Recall@50 сокращение списка без дополнительного ограничения невыгодно.
# Финальная проверка заново читает CSV как строки и проверяет все требования.
# Выбранный бустинг переобучается с теми же параметрами на OOF-признаках всех выбранных
# train/development/holdout/audit запросов. После этого локальные метрики не пересчитываем:
# они относятся к оценочной модели, а не к модели, уже видевшей отложенную разметку.

# %%
del history_model
gc.collect()
final_ranker = None
if ranker_choice["alpha"] > 0:
    final_training_queries = pd.concat([training_queries, validation, audit_queries], ignore_index=True)
    final_training_queries = final_training_queries[[*QUERY_COLS, "query_norm", "context_key"]].drop_duplicates("query_norm").sort_values("context_key").reset_index(drop=True)
    final_model_path = CACHE / f"final_ranker_{fingerprint}.joblib"
    if USE_CACHE and final_model_path.exists():
        final_ranker = joblib.load(final_model_path)
    else:
        final_training_data = build_oof_training(history_all, final_training_queries, "final")
        final_ranker = HistGradientBoostingClassifier(learning_rate=RANKER_CONFIG["learning_rate"],
            max_iter=ranker_choice["iterations"], max_leaf_nodes=ranker_choice["leaves"], min_samples_leaf=30,
            l2_regularization=RANKER_CONFIG["l2_regularization"], max_bins=127, early_stopping=False, random_state=SEED)
        final_ranker.fit(final_training_data["X"], final_training_data["y"], sample_weight=final_training_data["weight"])
        save_cache(final_ranker, final_model_path)
        del final_training_data
        gc.collect()
    print("Final ranker queries:", len(final_training_queries))
final_history_model = HistorySignals(history_all)
benchmark_cache = CACHE / f"benchmark_features_{fingerprint}.joblib"
if USE_CACHE and benchmark_cache.exists():
    benchmark_features = joblib.load(benchmark_cache)
    print("Loaded benchmark features")
else:
    benchmark_features = retrieve_features(queries, final_history_model)
    save_cache(benchmark_features, benchmark_cache)
benchmark_predictions = selected_predictions(queries, benchmark_features, final_ranker, "benchmark")
answer = pd.DataFrame({"query_id": queries.query_id.astype(str),
                       "answer": [" ".join(ITEM_IDS[indices]) for indices in benchmark_predictions]})

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

validate_answer(answer, queries.query_id, ITEM_IDS)
answer.to_csv(ROOT / "answer.csv", index=False, encoding="utf-8", lineterminator="\n")
reloaded_answer = pd.read_csv(ROOT / "answer.csv", dtype=str, keep_default_na=False)
validate_answer(reloaded_answer, queries.query_id, ITEM_IDS)
assert reloaded_answer.equals(answer)
assert reloaded_answer.answer.str.split().str.len().eq(50).all()
# Повторный отбор из тех же признаков должен дать тот же CSV побайтово.
repeat = pd.DataFrame({"query_id": queries.query_id.astype(str), "answer": [
    " ".join(ITEM_IDS[indices]) for indices in selected_predictions(queries, benchmark_features, final_ranker, "benchmark")]})
expected_bytes = repeat.to_csv(index=False, lineterminator="\n").encode("utf-8")
assert expected_bytes == (ROOT / "answer.csv").read_bytes()
manifest = {"config": CONFIG, "ranker_config": RANKER_CONFIG, "ranker_choice": ranker_choice,
            "input_sha256": input_hashes, "fingerprint": fingerprint,
            "code_sha256": code_hash,
            "answer_sha256": sha256_file(ROOT / "answer.csv"), "metrics": metrics,
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                         "numpy": np.__version__, "sklearn": sklearn.__version__}}
(CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Saved:", ROOT / "answer.csv")
print("Validated:", len(answer), "queries, 50 unique IDs each")
print("SHA-256:", manifest["answer_sha256"])

# %% [markdown]
# ## 13. Что сдавать и как интерпретировать результат
#
# Файл для платформы — `answer.csv`. Код решения целиком находится в этом ноутбуке.
# Для воспроизведения нужны исходные три Parquet и окружение из `requirements.txt`.
# В `artifacts/learned-selection-v1/` сохраняются конфигурация, протокол валидации, результаты
# экспериментов, анализ ошибок и контрольные суммы. Кеш индексов необязателен.
#
# Использованы открытые библиотеки pandas, NumPy, SciPy, scikit-learn, PyArrow,
# Snowball stemmer и joblib. HistGradientBoostingClassifier обучается на CPU;
# нейросетевые веса в этой версии не используются.
# Алгоритмы: BM25, TF-IDF по символьным n-граммам, поиск похожих запросов,
# сглаженные географические признаки и подбор весов на development.
#
# Источники алгоритмов и API:
# - [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
# - [BM25: параметры similarity](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)
# - [Snowball: русский stemmer](https://snowballstem.org/algorithms/russian/stemmer.html)
# - [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
#
# Ограничения: неполная поведенческая разметка, смещение валидации к пересечению
# корпусов, отсутствие семантического энкодера и ограничения длины текстов.
# Следующее улучшение нужно выбирать по сохранённому анализу ошибок и полноте пула.
#
# ### Результаты baseline v0.1.0 для сравнения
#
# На development: BM25 полного текста — **0.4141**, лучший вариант без географии —
# **0.4231**, гибрид с географией без подкатегорий — **0.8667**, выбранный гибрид —
# **0.8671**. Основной измеренный прирост связан с географией; вклад подкатегории
# в этом сравнении очень мал и пока не доказывает устойчивого улучшения.
#
# На holdout выбранный вариант получил **0.8912** (bootstrap 95%: **0.8683–0.9136**),
# полнота объединённого пула — **0.9714**. Полностью найдены положительные объявления
# для 621 из 700 запросов; частично — для 5; ни одного — для 74.
# 20 положительных объявлений потеряны до общего пула, 59 — при отборе 50 кандидатов.
#
# Для запросов, у которых все положительные объявления в той же локации, Recall@50
# равен **0.9156** (571 запрос); для остальных — **0.7829** (129 запросов).
# Это показывает ограничение географической модели, а не основание жёстко исключать
# другие города. Примеры разного словаря: «украшение помещений» → «Оформление входных
# групп, витрин, интерьеров»; многозначности: «аренда гуся» → объявление о фотосессии
# с животными. Эти примеры найдены в локальной разметке, не в скрытом benchmark.
#
# Меры baseline против ошибок: символьный поиск для вариантов написания, объединение
# нескольких источников, мягкая география. Ниже — результат добавленного отбора.
#
# ### Обучаемый отбор v0.2.0
#
# | Срез | Baseline Recall@50 | Обучаемый отбор |
# |---|---:|---:|
# | Development, 700 | 0.867143 | **0.904286** |
# | Прежний holdout, 700 | 0.891190 | **0.917857** |
# | Новый audit, 600 | 0.898333 | **0.923333** |
#
# Выбрана комбинация baseline и HistGradientBoosting (RRF 50/50): 120 деревьев,
# до 15 листьев. На audit прирост **+2.50 п.п.**, парный bootstrap 95%:
# **[+0.67; +4.33] п.п.** Улучшены 26 запросов, ухудшены 10. Audit не использовался
# для выбора конфигурации. Повторное использование старого holdout явно отмечено.
#
# Оценочная модель обучалась на 631 098 парах (4 058 positives), подготовленных
# из 4 000 запросов. У 81 запроса ни одного positive в пуле: такие запросы не дают
# обучающей группы, но в оценках Recall@50 потерянные positives остаются в знаменателе.
# Финальная модель обучена на 945 765 парах из 6 000 запросов после фиксации параметров.
# Для итоговой оценки используется модель до включения отложенной разметки в обучение.
#
# На старом holdout потери на отборе уменьшились с 59 до 38 объявлений; 20 по-прежнему
# отсутствуют в пуле. Исправлены, например, запросы «услуги фрезеровщика»,
# «массаж после кесарево», «электроскутер». Есть и регрессии: улучшились 33 запроса,
# ухудшились 12. Это результат всего набора признаков; отдельная причинная роль
# рейтинга или цены без ablation не утверждается.
#
# Следующий отдельный эксперимент — семантический поиск с измерением полноты пула
# и переобучением отбора на изменившихся кандидатах. `search_category` не меняли.
