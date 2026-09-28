"""Learn document-field scores on exactly the expanded v10 training examples.

Direct cosine blending can fail even when a signal helps conditionally on
geography or service type. This controlled extra-feature model keeps every v10
row, label and weight. Only frozen document-field similarities are appended;
sample-dependent ranks are avoided because training groups are sampled.
"""
from pathlib import Path
import ast
import gc
import hashlib
import json
import time

FIELD_RANK_DRIVER=Path(__file__).resolve()
source=FIELD_RANK_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(FIELD_RANK_DRIVER)
from activate_warm_history import activate
activate(globals())
FIELD_RANK_CACHE=ROOT/'artifacts/field-ranker-v10';FIELD_RANK_CACHE.mkdir(exist_ok=True)
FIELD_RANK_FP=hashlib.sha256((V10_FP+sha256_file(FIELD_RANK_DRIVER)+
    sha256_file(V10_CACHE/'selection.json')).encode()).hexdigest()[:16]
FIELD_RANK_NAMES=[f'{name}_{suffix}' for name in ['service_fields','long_description']
    for suffix in ['query_cosine','query_cosine_delta','query_filter_cosine','query_filter_delta']]
FIELD_DOCUMENTS=None


def prepare_fields():
    manifest=json.loads((ROOT/'artifacts/semantic-fields-v10/vectors.json').read_text())
    return {name:np.load(ROOT/manifest[name]['path'],mmap_mode='r') for name in ['service_fields','long_description']}


def field_features(query,ids,base):
    teacher=semantic_index.queries[semantic_index.query_to_row[query.query_norm]]
    contextual=CONTEXT_VECTORS[query_filter_text(query.query_norm,query.search_infm_params_text)]
    result=[]
    for name in ['service_fields','long_description']:
        docs=FIELD_DOCUMENTS[name][ids].astype(np.float64)
        plain=(docs@teacher.astype(np.float64)).astype(np.float32)
        filtered=(docs@contextual.astype(np.float64)).astype(np.float32)
        result.extend([plain,plain-base[:,32],filtered,filtered-base[:,72]])
    return np.column_stack(result).astype(np.float32)


def field_training(stage):
    selection=json.loads((V10_CACHE/'selection.json').read_text())
    warm=selection['winner']['variant'].startswith('warm_')
    data=mixed_training(stage) if warm else load_quality_training(stage)
    path=FIELD_RANK_CACHE/f'{stage}_X_{FIELD_RANK_FP}.npy'
    if not path.exists():
        frame=frame_for_stage(stage)[0].set_index('context_key')
        temporary=path.with_suffix('.npy.tmp')
        base_width=data['X'].shape[1]
        target=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(data['y']),base_width+8))
        offset=0
        def groups():
            if warm:
                for values in mixed_groups(stage):yield values[:6]
            else:
                for _,path in ordered_paths(V10_CACHE,stage,V10_FP):
                    yield from joblib.load(path)
        for x,y,known,key,text,ids in groups():
            stop=offset+len(y)
            assert np.array_equal(data['y'][offset:stop],y)
            assert np.array_equal(data['X'][offset:stop],x,equal_nan=True)
            q=frame.loc[key]
            target[offset:stop,:base_width]=x
            if float(q.search_category)==0:target[offset:stop,70]=1.
            target[offset:stop,base_width:]=field_features(q,ids,x)
            offset=stop
            if offset%200000<len(y):print('Append field training',stage,offset,len(data['y']),flush=True)
        assert offset==len(data['y'])
        target.flush();del target;gc.collect()
        temporary.replace(path)
    data['X']=np.load(path,mmap_mode='r')
    return data


def fit_field(data,stage):
    path=FIELD_RANK_CACHE/f'{stage}_ranker_{FIELD_RANK_FP}.joblib'
    if path.exists():return joblib.load(path)
    # Same parameters as quality: this is an extra-feature comparison.
    model=lgb.LGBMRanker(n_estimators=500,num_leaves=31,learning_rate=.04,reg_lambda=15,
        max_bin=127,min_child_samples=80,random_state=SEED,n_jobs=8,verbosity=-1,
        deterministic=True,force_col_wise=True,lambdarank_truncation_level=55,label_gain=[0,1])
    start=time.perf_counter();print('Fit field ranker',stage,data['X'].shape,flush=True)
    model.fit(data['X'],data['y'],group=data['sizes'],sample_weight=np.repeat(data['group_weight'],data['sizes']))
    save_cache(model,path)
    path.with_suffix('.json').write_text(json.dumps({'seconds':round(time.perf_counter()-start,2),
        'model_sha256':sha256_file(path),'features':NEW_FEATURE_NAMES+
            (TRACE_NAMES if data['X'].shape[1]==94 else [])+FIELD_RANK_NAMES},indent=2),encoding='utf-8')
    return model


def evaluate_field_ranker(stage,mode,model):
    global QUALITY_BANK
    frame,records,base,reference,known=v9_pool(stage,mode)
    selection=json.loads((V10_CACHE/'selection.json').read_text())
    reference=compatibility_boost(frame,records,reference,selection['category_boost'])
    history=evaluation_history(frame,mode,control if stage=='development' else development)
    QUALITY_BANK=(QualityEvidenceWithTrace if selection['winner']['variant'].startswith('warm_') else QualityEvidence)(history)
    del history;gc.collect()
    quality=extra_matrices(frame,records,base)
    for q,x in zip(frame.itertuples(index=False),quality):
        if float(q.search_category)==0:x[:,70]=1.
    winner=selection['winner'];name=winner['variant']
    if name in MODEL_RECIPES:
        recipe=MODEL_RECIPES[name]
        ranker=joblib.load(V10_CACHE/f'evaluation_{name}_{V10_FP}.joblib')
        score=predict_scores(ranker,[x[:,recipe['columns']] for x in quality],recipe['trees'])
        reference=blend_scores(score,reference,winner['weight']);del ranker,score
    augmented=[np.column_stack([x,field_features(q,ids,x)]).astype(np.float32)
        for q,(ids,_),x in zip(frame.itertuples(index=False),records,quality)]
    prediction=predict_scores(model,augmented,500)
    truth=labels_from_gold(gold,frame)
    values={0.:per_query_recall(top50(records,reference),truth)}
    for weight in [.15,.3,.5,.75,1.]:
        values[weight]=per_query_recall(top50(records,blend_scores(prediction,reference,weight)),truth)
    del records,base,quality,augmented,prediction,reference;gc.collect()
    return values,known


def compare_field_ranker():
    global FIELD_DOCUMENTS,CONTEXT_VECTORS
    FIELD_DOCUMENTS=prepare_fields()
    CONTEXT_VECTORS,_=contextual_vectors()
    data=field_training('evaluation');model=fit_field(data,'evaluation')
    del data;gc.collect()
    a,_=evaluate_field_ranker('development','unseen_text',model)
    b,known=evaluate_field_ranker('development','held_context',model)
    rows=[dict(weight=w,**matched_metrics(a[w],b[w],known)) for w in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(FIELD_RANK_CACHE/'development.csv',index=False,lineterminator='\n')
    baseline=next(r for r in rows if r['weight']==0.)
    winner=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)
        &(table.held_macro>=baseline['held_macro']-.0005)].iloc[0].to_dict()
    report={'fingerprint':FIELD_RANK_FP,'winner':winner,'baseline':baseline,
        'control_used_for_selection':False,'feature_names':NEW_FEATURE_NAMES+
            (TRACE_NAMES if json.loads((V10_CACHE/'selection.json').read_text())['winner']['variant'].startswith('warm_') else [])+FIELD_RANK_NAMES,
        'limitations':['Reused development/control; inherited auxiliary-prior limitations.']}
    (FIELD_RANK_CACHE/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Field ranker comparison',table.to_string(index=False),flush=True)
    c,_=evaluate_field_ranker('control','unseen_text',model)
    (FIELD_RANK_CACHE/'control.json').write_text(json.dumps({'baseline_recall50':float(c[0.].mean()),
        'selected_recall50':float(c[winner['weight']].mean()),'used_for_selection':False},indent=2),encoding='utf-8')


if __name__=='__main__':compare_field_ranker()
