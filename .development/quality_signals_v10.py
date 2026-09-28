"""Additional local signals from permitted history and frozen E5 vectors.

The caller must remove held query interactions before constructing the bank.
Query/filter vectors use an unsupervised, unchanged document encoder. There is
no benchmark label, query ID, special-case answer or external API in this module.
"""
import numpy as np
import pandas as pd

QUALITY_NAMES = ['search_category_matches_item', 'item_is_service_category',
    'query_filter_cosine', 'query_filter_cosine_delta', 'query_filter_log_rank',
    'item_history_log_count', 'item_history_log_query_count',
    'item_history_location_log_count', 'service_geo_transition_probability',
    'service_geo_log_support', 'service_same_location_fraction', 'delivery_search']


def query_filter_text(query, filters):
    """Stable text identity, independent of location, category and test IDs."""
    filters=' '.join(str(filters).split())
    return str(query) if not filters else str(query)+'; условия поиска: '+filters


class QualityEvidence:
    def __init__(self, history):
        self.item_counts=history.groupby('item_id').size().to_dict()
        self.item_queries=history.groupby('item_id').query_norm.nunique().to_dict()
        self.item_locations=history.groupby(['search_location_id','item_id']).size().to_dict()
        self.service_support=history.groupby(['search_location_id','item_microcat_id']).size().to_dict()
        transition=history.groupby(['search_location_id','item_microcat_id','item_location_id']).size()
        self.service_transitions={key:float(value/self.service_support[key[:2]]) for key,value in transition.items()}
        same=history.search_location_id.eq(history.item_location_id)
        self.service_locality=history.assign(same=same).groupby('item_microcat_id').same.mean().to_dict()

    def features(self, query, ids, item_ids, item_categories, item_microcats,
                 item_locations, item_vectors, contextual_vector, base_cosine, rankdata):
        document_ids=item_ids[ids]
        location=int(query.search_location_id)
        microcats=item_microcats[ids]
        cosine=(item_vectors[ids].astype(np.float64)@contextual_vector.astype(np.float64)).astype(np.float32)
        same_category=(float(query.search_category)>0)&(item_categories[ids]==float(query.search_category))
        rows=np.column_stack([
            same_category, item_categories[ids]==114, cosine, cosine-base_cosine,
            np.log1p(rankdata(-cosine,method='min')),
            np.log1p([self.item_counts.get(v,0) for v in document_ids]),
            np.log1p([self.item_queries.get(v,0) for v in document_ids]),
            np.log1p([self.item_locations.get((location,v),0) for v in document_ids]),
            [self.service_transitions.get((location,int(m),int(loc)),0.) for m,loc in zip(microcats,item_locations[ids])],
            np.log1p([self.service_support.get((location,int(m)),0) for m in microcats]),
            [self.service_locality.get(int(m),np.nan) for m in microcats],
            np.full(len(ids),float(query.search_is_delivery_search))])
        assert rows.shape[1]==len(QUALITY_NAMES) and not np.isinf(rows).any()
        return rows.astype(np.float32)
