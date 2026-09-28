"""Frozen v10 inference adapter used by the delivered notebook.

The adapter reads models, permitted training aggregates and deterministic
unsupervised query embeddings. SHA checks bind those dependencies to a manifest.
"""
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from quality_signals_v10 import query_filter_text


def adjust_quality_scores(root,manifest,queries,items,records,matrices,scores,
                          item_vectors,sha256_file,blend_scores,predict_scores):
    for key in ['history_bank','contextual_vectors','ranker','warm_aux_ranker']:
        if manifest.get(key):assert sha256_file(root/manifest[key])==manifest[key+'_sha256']
    for filename,digest in manifest['module_sha256'].items():assert sha256_file(root/filename)==digest
    category=pd.to_numeric(items.item_category_id,errors='coerce').to_numpy()
    strength=float(manifest['category_boost'])
    reference=[s+strength*((float(q.search_category)>0)&(category[ids]==float(q.search_category)))
        for q,(ids,_),s in zip(queries.itertuples(index=False),records,scores)]
    if not manifest.get('ranker') and not manifest.get('field_ranker') and not manifest.get('bge_kind') and not manifest.get('warm_aux_ranker'):return reference
    bank=joblib.load(root/manifest['history_bank'])
    bundle=joblib.load(root/manifest['contextual_vectors'])
    contextual=dict(zip(bundle['texts'],bundle['vectors']))
    item_ids=items.item_id.astype(str).to_numpy()
    micros=items.item_microcat_id.to_numpy();locations=items.item_location_id.to_numpy()
    features=[];full_features=[]
    for q,(ids,_),base in zip(queries.itertuples(index=False),records,matrices):
        vector=contextual[query_filter_text(q.query_norm,q.search_infm_params_text)]
        extra=bank.features(q,ids,item_ids,category,micros,locations,item_vectors,vector,base[:,32],rankdata)
        # Unspecified category permits every item; it is not incompatibility.
        if float(q.search_category)==0:extra[:,0]=1.
        complete=np.column_stack([base,extra]).astype(np.float32)
        full_features.append(complete)
        features.append(complete[:,:int(manifest.get('primary_feature_width',complete.shape[1]))])
    if manifest.get('ranker'):
        model=joblib.load(root/manifest['ranker'])
        assert model.n_features_in_==len(manifest['columns'])
        prediction=predict_scores(model,[x[:,manifest['columns']] for x in features],int(manifest['trees']))
        reference=blend_scores(prediction,reference,float(manifest['weight']))
    if manifest.get('field_documents'):
        assert sha256_file(root/manifest['field_teacher_vectors'])==manifest['field_teacher_vectors_sha256']
        teachers=joblib.load(root/manifest['field_teacher_vectors'])
        teacher_lookup=dict(zip(teachers['texts'],teachers['vectors']))
        field_documents={}
        for name,details in manifest['field_documents'].items():
            assert sha256_file(root/details['path'])==details['sha256']
            field_documents[name]=np.load(root/details['path'],mmap_mode='r')
        augmented=[]
        for q,(ids,_),x in zip(queries.itertuples(index=False),records,features):
            teacher=teacher_lookup[q.query_norm]
            filtered=contextual[query_filter_text(q.query_norm,q.search_infm_params_text)]
            columns=[]
            for name in ['service_fields','long_description']:
                docs=field_documents[name][ids].astype(np.float64)
                a=(docs@teacher.astype(np.float64)).astype(np.float32)
                b=(docs@filtered.astype(np.float64)).astype(np.float32)
                columns.extend([a,a-x[:,32],b,b-x[:,72]])
            augmented.append(np.column_stack([x,np.column_stack(columns)]).astype(np.float32))
        if manifest.get('field_ranker'):
            assert sha256_file(root/manifest['field_ranker'])==manifest['field_ranker_sha256']
            model=joblib.load(root/manifest['field_ranker'])
            assert model.n_features_in_==augmented[0].shape[1]
            prediction=predict_scores(model,augmented,500)
            reference=blend_scores(prediction,reference,float(manifest['field_weight']))
        features=augmented
    if manifest.get('bge_kind'):
        document_info=manifest['bge_documents']
        assert sha256_file(root/document_info['path'])==document_info['sha256']
        assert sha256_file(root/manifest['bge_queries'])==manifest['bge_queries_sha256']
        documents=np.load(root/document_info['path'],mmap_mode='r')
        vectors=joblib.load(root/manifest['bge_queries'])
        lookup=dict(zip(vectors['texts'],vectors['vectors']))
        extras=[]
        for q,(ids,_),x in zip(queries.itertuples(index=False),records,features):
            docs=documents[ids].astype(np.float64)
            a=(docs@lookup[q.query_norm].astype(np.float64)).astype(np.float32)
            text=query_filter_text(q.query_norm,q.search_infm_params_text)
            b=(docs@lookup[text].astype(np.float64)).astype(np.float32)
            extras.append(np.column_stack([a,a-x[:,32],b,b-x[:,72]]).astype(np.float32))
        if manifest['bge_kind']=='ranker':
            assert sha256_file(root/manifest['bge_ranker'])==manifest['bge_ranker_sha256']
            model=joblib.load(root/manifest['bge_ranker'])
            augmented=[np.column_stack([x,extra]).astype(np.float32) for x,extra in zip(features,extras)]
            assert model.n_features_in_==augmented[0].shape[1]
            prediction=predict_scores(model,augmented,int(manifest['bge_trees']))
        else:
            # Same geography-aware transform as semantic_affinity in training.
            prediction=[np.exp((extra[:,0]-1)/float(manifest['bge_temperature']))*(.03+.97*raw[:,4])
                for extra,(_,raw) in zip(extras,records)]
        reference=blend_scores(prediction,reference,float(manifest['bge_weight']))
    if manifest.get('warm_aux_ranker'):
        model=joblib.load(root/manifest['warm_aux_ranker'])
        assert model.n_features_in_==86 and full_features[0].shape[1]==86
        prediction=predict_scores(model,full_features,int(manifest['warm_aux_trees']))
        reference=blend_scores(prediction,reference,float(manifest['warm_aux_weight']))
    return reference
