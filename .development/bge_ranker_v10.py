"""Compare frozen BGE-M3 directly and as features on identical training pairs.

Upstream v10 choices are fixed before this experiment. BGE has no label-trained
encoder. Both query-only and query-plus-filter similarities are available.
Only development chooses a small blend grid; control follows that choice.
"""
from pathlib import Path
import ast
import gc
import hashlib
import json
import time

BGE_DRIVER=Path(__file__).resolve()
source=BGE_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(BGE_DRIVER)
from activate_warm_history import activate
activate(globals())
BGE_CACHE=ROOT/'artifacts/bge-m3-v10';BGE_CACHE.mkdir(exist_ok=True)
BGE_DOCUMENTS=None;BGE_QUERIES=None
BGE_NAMES=['bge_query_cosine','bge_query_delta_e5','bge_query_filter_cosine','bge_query_filter_delta_e5']
BGE_FP=hashlib.sha256((V10_FP+sha256_file(BGE_DRIVER)+sha256_file(V10_CACHE/'selection.json')+
    sha256_file(ROOT/'artifacts/field-ranker-v10/selection.json')+sha256_file(BGE_CACHE/'vectors.json')).encode()).hexdigest()[:16]


def install_field_functions():
    """Share the numeric field producer, without a second large bootstrap."""
    source=ROOT/'.development/field_ranker_v10.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),globals())
    selection=json.loads((ROOT/'artifacts/field-ranker-v10/selection.json').read_text())
    globals()['FIELD_RANK_DRIVER']=source
    globals()['FIELD_RANK_CACHE']=ROOT/'artifacts/field-ranker-v10'
    globals()['FIELD_RANK_FP']=selection['fingerprint']
    globals()['FIELD_RANK_NAMES']=selection['feature_names'][-8:]
    globals()['FIELD_DOCUMENTS']=prepare_fields() if selection['winner']['weight']>0 else None
    return selection


def prepare_bge():
    manifest=json.loads((BGE_CACHE/'vectors.json').read_text())
    documents=np.load(ROOT/manifest['documents']['path'],mmap_mode='r')
    texts=joblib.load(ROOT/manifest['query_texts']['path'])
    vectors=np.load(ROOT/manifest['queries']['path'],mmap_mode='r')
    return documents,dict(zip(texts,vectors))


def bge_features(query,ids,x):
    documents=BGE_DOCUMENTS[ids].astype(np.float64)
    a=(documents@BGE_QUERIES[query.query_norm].astype(np.float64)).astype(np.float32)
    text=query_filter_text(query.query_norm,query.search_infm_params_text)
    b=(documents@BGE_QUERIES[text].astype(np.float64)).astype(np.float32)
    return np.column_stack([a,a-x[:,32],b,b-x[:,72]]).astype(np.float32)


def bge_training(stage):
    primary=json.loads((V10_CACHE/'selection.json').read_text())
    warm=primary['winner']['variant'].startswith('warm_')
    field=json.loads((ROOT/'artifacts/field-ranker-v10/selection.json').read_text())
    use_field=field['winner']['weight']>0
    data=field_training(stage) if use_field else mixed_training(stage) if warm else load_quality_training(stage)
    path=BGE_CACHE/f'{stage}_X_{BGE_FP}.npy'
    if not path.exists():
        frame=frame_for_stage(stage)[0].set_index('context_key')
        temporary=path.with_suffix('.npy.partial')
        width=data['X'].shape[1]
        target=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(len(data['y']),width+4))
        def groups():
            if warm:
                for values in mixed_groups(stage):yield values[:6]
            else:
                for _,path in ordered_paths(V10_CACHE,stage,V10_FP):yield from joblib.load(path)
        offset=0
        for x,y,known,key,text,ids in groups():
            q=frame.loc[key]
            if use_field:x=np.column_stack([x,field_features(q,ids,x)]).astype(np.float32)
            stop=offset+len(y)
            # The actual unspecified-category correction is deterministic.
            if float(q.search_category)==0:x[:,70]=1.
            expected=data['X'][offset:stop].copy()
            if float(q.search_category)==0:expected[:,70]=1.
            assert np.array_equal(expected,x,equal_nan=True)
            assert np.array_equal(data['y'][offset:stop],y)
            target[offset:stop,:width]=x
            target[offset:stop,width:]=bge_features(q,ids,x)
            offset=stop
            if offset%500000<len(y):print('Append BGE training',stage,offset,len(data['y']),flush=True)
        assert offset==len(data['y'])
        target.flush();del target;gc.collect();temporary.replace(path)
    data['X']=np.load(path,mmap_mode='r')
    return data


def bge_model_path(stage,variant='compact'):
    suffix='ranker' if stage=='final' or variant=='compact' else 'deeper'
    return BGE_CACHE/f'{stage}_{suffix}_{BGE_FP}.joblib'


def fit_bge(data,stage,variant=None):
    if variant is None:
        variant=json.loads((BGE_CACHE/'selection.json').read_text())['winner']['variant'] if stage=='final' else 'compact'
    assert variant in {'compact','deep'}
    path=bge_model_path(stage,variant)
    if path.exists():return joblib.load(path)
    trees,leaves,min_child=(600,63,150) if variant=='deep' else (400,31,80)
    model=lgb.LGBMRanker(n_estimators=trees,num_leaves=leaves,learning_rate=.04,reg_lambda=15,
        max_bin=127,min_child_samples=min_child,random_state=SEED,n_jobs=8,verbosity=-1,
        deterministic=True,force_col_wise=True,lambdarank_truncation_level=55,label_gain=[0,1])
    started=time.perf_counter();print('Fit BGE ranker',stage,variant,data['X'].shape,flush=True)
    model.fit(data['X'],data['y'],group=data['sizes'],sample_weight=np.repeat(data['group_weight'],data['sizes']))
    save_cache(model,path)
    path.with_suffix('.json').write_text(json.dumps({'seconds':round(time.perf_counter()-started,2),
        'model_sha256':sha256_file(path),'features':data['X'].shape[1],'bge_features':BGE_NAMES,
        'variant':variant,'trees':trees,'leaves':leaves,'min_child_samples':min_child},indent=2),encoding='utf-8')
    return model


def audit_bge_candidate_recall():
    """Measure recall ceiling before deciding whether new pools are necessary.

    This is not a submission variant: every ranker comparison below uses the
    same frozen v9 pool. The diagnostic separates retrieval misses from ranking
    errors and does not claim that extra recalled items reach the final top-50.
    """
    transpose=BGE_DOCUMENTS.astype(np.float64).T
    rows=[]
    for mode in ['unseen_text','held_context']:
        frame,records,base,reference,known=v9_pool('development',mode)
        history=evaluation_history(frame,mode,control)
        retrieval_history=V5History(history);del history;gc.collect()
        truth=labels_from_gold(gold,frame)
        baseline=per_query_recall([r[0] for r in records],truth)
        expanded=[];additional=[];block=None
        for row,q in enumerate(frame.itertuples(index=False)):
            if row%32==0:
                texts=frame.query_norm.iloc[row:row+32]
                block=(np.stack([BGE_QUERIES[t] for t in texts]).astype(np.float64)@transpose).astype(np.float32)
            cosine=block[row%32]
            geo=retrieval_history.geography(int(q.search_location_id))
            ids=np.union1d(stable_topk(cosine,500),stable_topk(semantic_affinity(cosine,geo),500))
            union=np.union1d(records[row][0],ids)
            expanded.append(union);additional.append(len(union)-len(records[row][0]))
        after=per_query_recall(expanded,truth)
        rows.append({'mode':mode,'original_pool_recall':float(baseline.mean()),
            'with_bge_pool_recall':float(after.mean()),'improved_queries':int((after>baseline).sum()),
            'mean_extra_candidates':float(np.mean(additional))})
        del records,base,reference,retrieval_history,expanded;gc.collect()
    del transpose;gc.collect()
    report={'rows':rows,'diagnostic_only':True,'submission_pool_unchanged':True,
        'limitation':'An increased recall ceiling is not measured Recall@50 after selection.'}
    (BGE_CACHE/'candidate_recall_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('BGE candidate-recall ceiling',json.dumps(report),flush=True)


def evaluate_bge(stage,mode,models):
    global QUALITY_BANK
    frame,records,base,reference,known=v9_pool(stage,mode)
    primary=json.loads((V10_CACHE/'selection.json').read_text())
    reference=compatibility_boost(frame,records,reference,primary['category_boost'])
    history=evaluation_history(frame,mode,control if stage=='development' else development)
    QUALITY_BANK=(QualityEvidenceWithTrace if primary['winner']['variant'].startswith('warm_') else QualityEvidence)(history)
    del history;gc.collect()
    x=extra_matrices(frame,records,base)
    for q,values in zip(frame.itertuples(index=False),x):
        if float(q.search_category)==0:values[:,70]=1.
    name=primary['winner']['variant']
    if name in MODEL_RECIPES:
        recipe=MODEL_RECIPES[name]
        ranker=joblib.load(V10_CACHE/f'evaluation_{name}_{V10_FP}.joblib')
        prediction=predict_scores(ranker,[v[:,recipe['columns']] for v in x],recipe['trees'])
        reference=blend_scores(prediction,reference,primary['winner']['weight'])
        del ranker,prediction
    field=json.loads((ROOT/'artifacts/field-ranker-v10/selection.json').read_text())
    if field['winner']['weight']>0:
        x=[np.column_stack([values,field_features(q,ids,values)]).astype(np.float32)
            for q,(ids,_),values in zip(frame.itertuples(index=False),records,x)]
        ranker=joblib.load(FIELD_RANK_CACHE/f'evaluation_ranker_{FIELD_RANK_FP}.joblib')
        reference=blend_scores(predict_scores(ranker,x,500),reference,field['winner']['weight'])
        del ranker
    bge=[bge_features(q,ids,values) for q,(ids,_),values in zip(frame.itertuples(index=False),records,x)]
    augmented=[np.column_stack([values,extra]).astype(np.float32) for values,extra in zip(x,bge)]
    direct=[semantic_affinity(extra[:,0],raw[:,4]) for extra,(_,raw) in zip(bge,records)]
    truth=labels_from_gold(gold,frame)
    values={('baseline','none',0.):per_query_recall(top50(records,reference),truth)}
    for weight in [.025,.05,.1]:
        values[('direct','none',weight)]=per_query_recall(top50(records,blend_scores(direct,reference,weight)),truth)
    for variant,model in models.items():
        predicted=predict_scores(model,augmented,int(model.n_estimators))
        for weight in [.15,.3,.5,.75,1.]:
            values[('ranker',variant,weight)]=per_query_recall(top50(records,blend_scores(predicted,reference,weight)),truth)
        del predicted
    del records,base,x,bge,augmented,direct,reference;gc.collect()
    return values,known


def compare():
    global BGE_DOCUMENTS,BGE_QUERIES,CONTEXT_VECTORS
    install_field_functions();BGE_DOCUMENTS,BGE_QUERIES=prepare_bge()
    CONTEXT_VECTORS,_=contextual_vectors()
    audit_bge_candidate_recall()
    data=bge_training('evaluation')
    models={variant:fit_bge(data,'evaluation',variant) for variant in ['compact','deep']}
    del data;gc.collect()
    a,_=evaluate_bge('development','unseen_text',models)
    b,known=evaluate_bge('development','held_context',models)
    rows=[dict(kind=kind,variant=variant,weight=weight,**matched_metrics(a[(kind,variant,weight)],b[(kind,variant,weight)],known)) for kind,variant,weight in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(BGE_CACHE/'development.csv',index=False,lineterminator='\n')
    baseline=next(row for row in rows if row['kind']=='baseline')
    eligible=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)&(table.held_macro>=baseline['held_macro']-.0005)]
    winner=eligible.iloc[0].to_dict()
    selection={'fingerprint':BGE_FP,'winner':winner,'baseline':baseline,'control_used_for_selection':False,
        'trees':600 if winner['variant']=='deep' else 400,
        'model':'BAAI/bge-m3','source':'https://huggingface.co/BAAI/bge-m3','license':'MIT',
        'limitations':['Reused development/control and inherited auxiliary priors; noisy unchosen negatives.']}
    (BGE_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('BGE-M3 comparison',table.to_string(index=False),flush=True)
    chosen={winner['variant']:models[winner['variant']]} if winner['kind']=='ranker' else {}
    c,_=evaluate_bge('control','unseen_text',chosen)
    (BGE_CACHE/'control.json').write_text(json.dumps({'baseline_recall50':float(c[('baseline','none',0.)].mean()),
        'selected_recall50':float(c[(winner['kind'],winner['variant'],winner['weight'])].mean()),'used_for_selection':False},indent=2),encoding='utf-8')


if __name__=='__main__':compare()
