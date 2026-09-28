"""Independent v11 features from frozen BGE scores and item metadata.

Every corpus-relative BGE statistic uses the same 189,212-item corpus at train
and inference time. Metadata features use no click labels or query IDs.
"""
import re

import numpy as np
import pandas as pd


BGE_RELATIVE_NAMES = [
    'bge_global_log_rank', 'bge_filter_global_log_rank',
    'bge_margin_best', 'bge_margin_top50', 'bge_filter_margin_best',
    'bge_filter_margin_top50', 'bge_top512_normalized',
    'bge_filter_top512_normalized', 'bge_geo_affinity',
    'bge_filter_geo_affinity', 'bge_filter_effect',
]

STRUCTURED_NAMES = [
    'filter_service_match', 'filter_type_match',
    'log_item_location_density', 'log_item_micro_location_density',
    'micro_location_share', 'log_query_micro_location_density',
    'item_remote_service', 'item_travels', 'item_does_not_travel',
    'travel_distance_relative',
]


def _extract(text, label, stops):
    """Read one flat exported parameter until the next known field label."""
    if not isinstance(text, str):
        return ''
    start = text.find(label)
    if start < 0:
        return ''
    start += len(label)
    end = len(text)
    for marker in stops:
        position = text.find(marker, start)
        if 0 <= position < end:
            end = position
    return ' '.join(text[start:end].strip().lower().split())


QUERY_STOPS = [' Вид услуги ', ' Тип услуги ', ' Предмет или специальность ',
               ' Рейтинг пользователя ', ' Цена ', ' Стоимость ',
               ' Место оказания услуг ']
ITEM_STOPS = [' Вид услуги ', ' Тип услуги ', ' Место оказания услуг ',
              ' Тип стоимости ', ' График работы ', ' Где вы оказываете услуги ',
              ' Начальная цена ', ' Стоимость ', ' Опыт работы ',
              ' Работа по договору ', ' Гарантия ', ' Выезд ',
              ' Куда выезжаете ', ' Груз ', ' Услуга ']
REMOTE = re.compile(r'\b(?:дистанционн\w*|удал[её]нн\w*|онлайн[ -]услуг\w*|'
                    r'работа[а-я ]*онлайн|как вы работаете онлайн)\b', re.IGNORECASE)


class ItemMetadata:
    def __init__(self, items):
        params = items.item_infm_params_text.fillna('').astype(str).to_numpy()
        self.service = np.asarray([_extract(s, 'Вид услуги ', ITEM_STOPS)
                                   for s in params], dtype=object)
        self.type = np.asarray([_extract(s, 'Тип услуги ', ITEM_STOPS)
                                for s in params], dtype=object)
        self.remote = np.asarray([bool(REMOTE.search(s[:2000])) for s in params],
                                 dtype=np.float32)
        self.travels = np.asarray(['Куда выезжаете' in s and 'Не выезжаю' not in s
                                   for s in params], dtype=np.float32)
        self.not_travel = np.asarray(['Не выезжаю' in s for s in params],
                                     dtype=np.float32)
        self.micro = pd.to_numeric(items.item_microcat_id, errors='coerce').fillna(-1).to_numpy(np.int64)
        self.location = pd.to_numeric(items.item_location_id, errors='coerce').fillna(-1).to_numpy(np.int64)
        loc_counts = pd.Series(self.location).value_counts().to_dict()
        pair_counts = pd.DataFrame({'micro': self.micro, 'location': self.location}).value_counts().to_dict()
        self.location_count = np.asarray([loc_counts[int(v)] for v in self.location], dtype=np.float32)
        self.pair_count = np.asarray([pair_counts[(int(m), int(l))]
                                      for m, l in zip(self.micro, self.location)], dtype=np.float32)
        self.pair_lookup = pair_counts

    def features(self, query, ids, base):
        n = len(ids)
        query_params = str(query.search_infm_params_text)
        q_service = _extract(query_params, 'Вид услуги ', QUERY_STOPS)
        q_type = _extract(query_params, 'Тип услуги ', QUERY_STOPS)
        service = np.full(n, np.nan, dtype=np.float32)
        typ = np.full(n, np.nan, dtype=np.float32)
        if q_service:
            service = (self.service[ids] == q_service).astype(np.float32)
            service[self.service[ids] == ''] = np.nan
        if q_type:
            typ = (self.type[ids] == q_type).astype(np.float32)
            typ[self.type[ids] == ''] = np.nan
        location = int(query.search_location_id)
        micros, reverse = np.unique(self.micro[ids], return_inverse=True)
        query_density = np.asarray([self.pair_lookup.get((int(m), location), 0)
                                    for m in micros], dtype=np.float32)[reverse]
        travel = self.travels[ids]
        # Both direct and history-imputed distances are allowed. Their missingness
        # and uncertainty already exist in the baseline feature matrix.
        distance = np.where(np.isfinite(base[:, 12]), base[:, 12], base[:, 51])
        result = np.column_stack([
            service, typ, np.log1p(self.location_count[ids]),
            np.log1p(self.pair_count[ids]),
            self.pair_count[ids] / np.maximum(self.location_count[ids], 1),
            np.log1p(query_density), self.remote[ids], travel,
            self.not_travel[ids], travel * distance,
        ]).astype(np.float32)
        assert result.shape == (n, len(STRUCTURED_NAMES))
        return result


class BgeReference:
    def __init__(self, texts, score_table):
        self.lookup = {text: row for row, text in enumerate(texts)}
        self.scores = score_table
        assert score_table.shape[1] == 512

    @staticmethod
    def _one(scores, table):
        thresholds = np.asarray(table, dtype=np.float32)
        ranks = np.searchsorted(-thresholds, -scores, side='left') + 1
        spread = max(float(np.std(thresholds)), 1e-3)
        return [np.log1p(ranks), scores-thresholds[0],
                scores-thresholds[49], (scores-float(np.mean(thresholds)))/spread]

    def features(self, query_text, filter_text, base):
        plain = base[:, -4]
        filtered = base[:, -2]
        a = self._one(plain, self.scores[self.lookup[query_text]])
        b = self._one(filtered, self.scores[self.lookup[filter_text]])
        return np.column_stack([a[0], b[0], *a[1:3], *b[1:3], a[3], b[3],
                                plain*base[:,4], filtered*base[:,4],
                                filtered-plain]).astype(np.float32)
