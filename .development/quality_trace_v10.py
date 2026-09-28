"""Exact query-item evidence from permitted training interactions only.

Repeats in other allowed contexts are legitimate history, not benchmark labels.
Unknown query texts receive zeros. The caller excludes own held contexts/texts.
"""
import numpy as np
from quality_signals_v10 import QualityEvidence

TRACE_NAMES=['query_item_history_log_count','query_location_item_history_log_count',
    'query_filter_item_history_log_count','query_item_history_probability']


def filter_key(value):return ' '.join(str(value).split())


class QueryItemEvidence:
    def __init__(self,history):
        self.query_items=history.groupby(['query_norm','item_id']).size().to_dict()
        self.query_locations=history.groupby(['query_norm','search_location_id','item_id']).size().to_dict()
        normalized=history.assign(filter_key=history.search_infm_params_text.map(filter_key))
        self.query_filters=normalized.groupby(['query_norm','filter_key','item_id']).size().to_dict()
        self.query_counts=history.groupby('query_norm').size().to_dict()

    def features(self,query,item_ids):
        text=query.query_norm;location=int(query.search_location_id);filters=filter_key(query.search_infm_params_text)
        count=np.array([self.query_items.get((text,item),0) for item in item_ids],np.float32)
        return np.column_stack([np.log1p(count),
            np.log1p([self.query_locations.get((text,location,item),0) for item in item_ids]),
            np.log1p([self.query_filters.get((text,filters,item),0) for item in item_ids]),
            count/max(float(self.query_counts.get(text,0)),1.)]).astype(np.float32)


class QualityEvidenceWithTrace(QualityEvidence):
    def __init__(self,history):
        super().__init__(history)
        self.query_item=QueryItemEvidence(history)

    def features(self,query,ids,item_ids,*args):
        base=super().features(query,ids,item_ids,*args)
        if float(query.search_category)==0:base[:,0]=1.
        return np.column_stack([base,self.query_item.features(query,item_ids[ids])]).astype(np.float32)
