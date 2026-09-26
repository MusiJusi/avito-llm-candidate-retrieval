# %% [markdown]
# # Кандидатогенерация объявлений услуг Авито
#
# **Цель:** вернуть 50 уникальных `item_id` для каждого запроса; метрика — macro Recall@50.
# Это воспроизводимая первая версия: поиск BM25 по словам и TF-IDF по символьным
# n-граммам, мягкий учёт географии и перенос информации о подкатегории из train.
# Все вычисления локальные. Внешние API и разметка benchmark не используются.
#
# **Запуск:** положить три исходных Parquet рядом с ноутбуком, установить
# `requirements.txt` в окружение его ядра и выполнить **Restart Kernel → Run All**.
# На текущем компьютере зависимости уже находятся в `.inspection_deps`.
# Ноутбук сам строит индексы и сохраняет `answer.csv`, отчёт и анализ ошибок.
# Кеш ускоряет повторный запуск; его можно удалить и пересчитать всё из исходных файлов.
#
# В этой версии специально нет зависимости от GPU. Эмбеддинги и обучаемый отбор
# будут отдельными экспериментами, сравниваемыми с данным baseline.

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
from threadpoolctl import threadpool_limits

SEED = 260926
CONFIG = {
    "seed": SEED, "validation_texts": 1400, "development_fraction": 0.5,
    "cold_item_fraction": 0.9, "description_chars": 4000,
    "params_chars": 1800, "word_features": 240_000,
    "char_features": 220_000, "retrieval_pool": 500,
    "version": "lexical-v1",
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
# ## 7. Отложенная оценка и анализ ошибок
#
# Конфигурация уже выбрана на development. Holdout используется для итоговой оценки,
# а не перебора весов. Доверительный интервал получаем bootstrap по запросам.
# В таблице ошибок различаем потерю при создании пула и при отборе 50 кандидатов.
# Для ручного анализа сохраняем текст запроса и заголовки пропущенных объявлений.

# %%
holdout_records = [validation_features[i] for i in holdout_indices]
holdout_labels = [labels[i] for i in holdout_indices]
holdout_predictions = predict_from_features(holdout_records, best_config)
holdout_recall = per_query_recall(holdout_predictions, holdout_labels)
pool_recall = per_query_recall([r[0] for r in holdout_records], holdout_labels)
boot_rng = np.random.default_rng(SEED)
bootstrap = np.mean(boot_rng.choice(holdout_recall, size=(2000, len(holdout_recall)), replace=True), axis=1)
metrics = {
    "protocol": "cold_query_text_90pct_purged_positive_items_full_benchmark_corpus",
    "development_queries": len(dev_indices), "holdout_queries": len(holdout_indices),
    "development_recall50": float(results.iloc[0].development_recall50),
    "holdout_recall50": float(holdout_recall.mean()),
    "holdout_recall50_bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
    "holdout_pool_recall": float(pool_recall.mean()),
    "mean_pool_size": float(np.mean([len(r[0]) for r in holdout_records])),
    "best_config": best_config,
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
# ## 8. Финальное обучение истории и ответ для benchmark
#
# После выбора параметров возвращаем всю обучающую историю, включая локальную
# валидацию: это допустимо, поскольку benchmark-разметка неизвестна и не используется.
# Текстовые индексы корпуса остаются теми же. Возвращаем ровно 50 кандидатов:
# при Recall@50 сокращение списка без дополнительного ограничения невыгодно.
# Финальная проверка заново читает CSV как строки и проверяет все требования.

# %%
del history_model
gc.collect()
final_history_model = HistorySignals(history_all)
benchmark_cache = CACHE / f"benchmark_features_{fingerprint}.joblib"
if USE_CACHE and benchmark_cache.exists():
    benchmark_features = joblib.load(benchmark_cache)
    print("Loaded benchmark features")
else:
    benchmark_features = retrieve_features(queries, final_history_model)
    save_cache(benchmark_features, benchmark_cache)
benchmark_predictions = predict_from_features(benchmark_features, best_config)
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
    " ".join(ITEM_IDS[indices]) for indices in predict_from_features(benchmark_features, best_config)]})
expected_bytes = repeat.to_csv(index=False, lineterminator="\n").encode("utf-8")
assert expected_bytes == (ROOT / "answer.csv").read_bytes()
manifest = {"config": CONFIG, "input_sha256": input_hashes, "fingerprint": fingerprint,
            "code_sha256": code_hash,
            "answer_sha256": sha256_file(ROOT / "answer.csv"), "metrics": metrics,
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                         "numpy": np.__version__, "sklearn": sklearn.__version__}}
(CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Saved:", ROOT / "answer.csv")
print("Validated:", len(answer), "queries, 50 unique IDs each")
print("SHA-256:", manifest["answer_sha256"])

# %% [markdown]
# ## 9. Что сдавать и как интерпретировать результат
#
# Файл для платформы — `answer.csv`. Код решения целиком находится в этом ноутбуке.
# Для воспроизведения нужны исходные три Parquet и окружение из `requirements.txt`.
# В `artifacts/lexical-v1/` сохраняются конфигурация, протокол валидации, результаты
# экспериментов, анализ ошибок и контрольные суммы. Кеш индексов необязателен.
#
# Использованы открытые библиотеки pandas, NumPy, SciPy, scikit-learn, PyArrow,
# Snowball stemmer и joblib. Нейросетевые веса в этой версии не используются.
# Алгоритмы: BM25, TF-IDF по символьным n-граммам, поиск похожих запросов,
# сглаженные географические признаки и подбор весов на development.
#
# Источники алгоритмов и API:
# - [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
# - [BM25: параметры similarity](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)
# - [Snowball: русский stemmer](https://snowballstem.org/algorithms/russian/stemmer.html)
#
# Ограничения: неполная поведенческая разметка, смещение валидации к пересечению
# корпусов, отсутствие семантического энкодера и ограничения длины текстов.
# Следующее улучшение нужно выбирать по сохранённому анализу ошибок и полноте пула.
#
# ### Результаты первого эксперимента
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
# Текущие меры против ошибок: символьный поиск для вариантов написания, объединение
# нескольких источников, мягкая география. Нерешённые ошибки сохраняются в отчёте;
# следующий эксперимент — обучаемый отбор из пула и семантический источник кандидатов.
