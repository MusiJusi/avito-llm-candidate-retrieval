"""Portable v11 scoring adapter layered on the frozen v10 candidate pool.

It rebuilds exactly the v10 input columns from locally packaged weights, adds
the selected v11 block, and predicts in bounded batches. No online service is
called. The external notebook verifies the final answer hash after export.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quality_signals_v10 import query_filter_text
from rank_features_v11 import BgeReference, ItemMetadata


def adjust_v11_scores(root, v10_manifest, v11_manifest, queries, items,
                      records, context_matrices, old_scores, item_vectors,
                      sha256_file, blend_scores, predict_scores):
    """Return the chosen blend of v11 and verified v10 scores."""
    for key in ['ranker', 'reference']:
        path = root / v11_manifest[key]
        assert sha256_file(path) == v11_manifest[key+'_sha256']
    model = joblib.load(root / v11_manifest['ranker'])
    reference_bundle = joblib.load(root / v11_manifest['reference'])
    reference = BgeReference(reference_bundle['texts'], reference_bundle['scores'])
    block = v11_manifest['block']
    structured = block in {'structured', 'metadata'}
    metadata = ItemMetadata(items) if structured else None
    bank = joblib.load(root / v10_manifest['history_bank'])
    contextual_bundle = joblib.load(root / v10_manifest['contextual_vectors'])
    contextual = dict(zip(contextual_bundle['texts'], contextual_bundle['vectors']))
    field_manifest = v10_manifest['field_documents']
    field_docs = {name: np.load(root / details['path'], mmap_mode='r')
                  for name, details in field_manifest.items()}
    teacher_bundle = joblib.load(root / v10_manifest['field_teacher_vectors'])
    teacher_vectors = dict(zip(teacher_bundle['texts'], teacher_bundle['vectors']))
    bge_docs = np.load(root / v10_manifest['bge_documents']['path'], mmap_mode='r')
    bge_bundle = joblib.load(root / v10_manifest['bge_queries'])
    bge_queries = dict(zip(bge_bundle['texts'], bge_bundle['vectors']))
    assert float(v10_manifest['bge_weight']) == 1.0
    assert not v10_manifest.get('warm_aux_weight') and not v10_manifest.get('cross_weight')
    old_ranker = joblib.load(root / v10_manifest['bge_ranker'])
    categories = pd.to_numeric(items.item_category_id, errors='coerce').to_numpy()
    item_ids = items.item_id.astype(str).to_numpy()
    micros = items.item_microcat_id.to_numpy()
    locations = items.item_location_id.to_numpy()
    output = []
    for first in range(0, len(records), 32):
        matrices = []
        original_matrices = []
        for q, (ids, _), base in zip(
                queries.iloc[first:first+32].itertuples(index=False),
                records[first:first+32], context_matrices[first:first+32]):
            text = query_filter_text(q.query_norm, q.search_infm_params_text)
            contextual_vector = contextual[text]
            quality = bank.features(q, ids, item_ids, categories, micros,
                                    locations, item_vectors, contextual_vector,
                                    base[:,32], rankdata)
            if float(q.search_category) == 0:
                quality[:,0] = 1.
            complete = np.column_stack([base, quality]).astype(np.float32)
            x = complete[:, :int(v10_manifest.get('primary_feature_width',
                                                 complete.shape[1]))]
            columns = []
            teacher = teacher_vectors[q.query_norm]
            for name in ['service_fields', 'long_description']:
                docs = field_docs[name][ids].astype(np.float64)
                plain = (docs @ teacher.astype(np.float64)).astype(np.float32)
                filtered = (docs @ contextual_vector.astype(np.float64)).astype(np.float32)
                columns.extend([plain, plain-x[:,32], filtered, filtered-x[:,72]])
            x = np.column_stack([x, np.column_stack(columns)]).astype(np.float32)
            documents = bge_docs[ids].astype(np.float64)
            plain = (documents @ bge_queries[q.query_norm].astype(np.float64)).astype(np.float32)
            filtered = (documents @ bge_queries[text].astype(np.float64)).astype(np.float32)
            x = np.column_stack([x, plain, plain-x[:,32],
                                 filtered, filtered-x[:,72]]).astype(np.float32)
            assert x.shape[1] == 94
            assert not np.isinf(x).any()
            if first == 0:
                original_matrices.append(x)
            relative = reference.features(q.query_norm, text, x) if block in {'relative', 'structured'} else None
            extra = metadata.features(q, ids, x) if structured else None
            parts = [x]
            if relative is not None: parts.append(relative)
            if extra is not None: parts.append(extra)
            x = np.column_stack(parts).astype(np.float32)
            assert x.shape[1] == model.n_features_in_, (x.shape, model.n_features_in_)
            matrices.append(x)
        if first == 0:
            # v10's final score is exactly the BGE ranker output in its frozen
            # manifest. Check reconstructed inputs against that score.
            reconstructed = predict_scores(old_ranker, original_matrices,
                                           int(v10_manifest['bge_trees']))
            assert all(np.allclose(a, b, atol=1e-6, rtol=1e-6)
                       for a, b in zip(reconstructed, old_scores[:len(reconstructed)])), \
                'v10 scoring inputs changed while rebuilding v11 features'
        output.extend(predict_scores(model, matrices, int(v11_manifest['trees'])))
    return blend_scores(output, old_scores, float(v11_manifest['weight']))
