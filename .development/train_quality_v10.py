"""Expand OOF ranker coverage with soft category and contextual E5 signals.

All training contexts are eligible; chunks and memory-mapped arrays limit peak
RAM. The base ablation and extra-feature model use exactly the same examples.
Sampling retains all retrieved positives, broad v7 competitors and a random
tail. A separate text-held miner uses only unsupervised feature columns; its
training labels exclude the current text fold. v7 mining remains an explicitly
in-sample component, not misrepresented as independent OOF mining.
"""
from pathlib import Path
import ast
import gc
import json
import hashlib
import time
import argparse

V10_TRAIN_DRIVER=Path(__file__).resolve()
source=V10_TRAIN_DRIVER.with_name('quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(V10_TRAIN_DRIVER)
from quality_signals_v10 import QualityEvidence,QUALITY_NAMES,query_filter_text

V10_CONFIG={'evaluation_contexts':19031,'final_contexts':26556,'chunk':128,
    'trees':500,'leaves':31,'seed':SEED+1201,'oof_mined_negatives':120,
    'category_allowed':True,'contextual_encoder':'frozen original E5, query + filters',
    'selection_grid':[0.,.15,.3,.5,.75,1.]}
V10_FP=hashlib.sha256(json.dumps({'base':V9_FP,'config':V10_CONFIG,
    'source':sha256_file(V10_TRAIN_DRIVER),'signals':sha256_file(V10_TRAIN_DRIVER.with_name('quality_signals_v10.py'))},
    sort_keys=True).encode()).hexdigest()[:16]
NEW_FEATURE_NAMES=FEATURE_NAMES+QUALITY_NAMES
ITEM_MICROS=items.item_microcat_id.to_numpy()
QUALITY_BANK=None
CONTEXT_VECTORS=None
# Avoid history scores, learned encoder scores and sampling-source indicators.
MINER_COLUMNS=[0,1,2,3,11,12,*range(13,33)]


def contextual_vectors():
    frame=pd.concat([history_all[['query_norm','search_infm_params_text']],
        queries[['query_norm','search_infm_params_text']]],ignore_index=True).drop_duplicates()
    texts=sorted(set(query_filter_text(q,f) for q,f in frame.itertuples(index=False,name=None)))
    digest=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()[:16]
    path=V10_CACHE/f'query_filter_vectors_{digest}.joblib'
    if path.exists():bundle=joblib.load(path)
    else:
        vectors=semantic_index.encode(texts,'query: ')
        bundle={'texts':texts,'vectors':vectors,'encoder':'original frozen E5','max_length':SEMANTIC_CONFIG['max_length']}
        save_cache(bundle,path)
        semantic_index.encoder=semantic_index.tokenizer=None
        gc.collect()
        if DEVICE=='cuda':torch.cuda.empty_cache()
    return dict(zip(bundle['texts'],bundle['vectors'])),path


def extra_matrices(frame,records,matrices):
    output=[]
    for q,(ids,_),x in zip(frame.itertuples(index=False),records,matrices):
        vector=CONTEXT_VECTORS[query_filter_text(q.query_norm,q.search_infm_params_text)]
        extra=QUALITY_BANK.features(q,ids,ITEM_IDS,ITEM_CATEGORIES,ITEM_MICROS,ITEM_LOCS,
            semantic_index.items,vector,x[:,32],rankdata)
        output.append(np.column_stack([x,extra]).astype(np.float32))
    return output


def independent_miners(stage):
    """Text-held label isolation on pure lexical/metadata/frozen-E5 features.

    Pools were prepared with historical retrieval; this is not claimed to be a
    fully nested evaluation of retrieval. History-dependent score columns and
    supervised query encoders are expressly absent from the mining model.
    """
    result={}
    for held_fold in range(3):
        path=V10_CACHE/f'{stage}_textheld_miner_{held_fold}_{V10_FP}.joblib'
        if path.exists():result[held_fold]=joblib.load(path);continue
        xs=[];ys=[];sizes=[]
        for fold in range(3):
            if fold==held_fold:continue
            for chunk in sorted(V9_CACHE.glob(f'{stage}_unseen_text_{fold}_*_{V9_FP}.joblib')):
                for x,y,_ in joblib.load(chunk):
                    xs.append(x[:,MINER_COLUMNS]);ys.append(y);sizes.append(len(y))
        assert sizes,'Missing v9 chunks for independent mining'
        x=np.concatenate(xs);y=np.concatenate(ys)
        del xs,ys;gc.collect()
        miner=lgb.LGBMRanker(n_estimators=200,num_leaves=31,learning_rate=.05,
            reg_lambda=20,max_bin=127,min_child_samples=100,n_jobs=8,verbosity=-1,
            random_state=SEED,deterministic=True,force_col_wise=True,
            lambdarank_truncation_level=55,label_gain=[0,1])
        print('Fit text-held miner',stage,held_fold,x.shape,flush=True)
        miner.fit(x,y,group=sizes)
        save_cache(miner,path);result[held_fold]=miner
        del x,y;gc.collect()
    return result


def frame_for_stage(stage):
    if stage=='evaluation':return training,training_history,QR_CACHE,QR_FP
    frame=history_all[history_all.context_key.isin(gold)].drop_duplicates('context_key')
    frame=frame[[*QUERY_COLS,'query_norm','context_key']].sort_values('context_key').reset_index(drop=True)
    cache=ROOT/'artifacts/query-encoder-candidate'
    fp=json.loads((cache/'manifest.json').read_text())['fingerprint']
    return frame,history_all,cache,fp


def prepare_quality_training(stage):
    global LEARNED_LOOKUP,QUALITY_BANK
    manifest_path=V10_CACHE/f'{stage}_training_{V10_FP}.json'
    if manifest_path.exists():return load_quality_training(stage)
    frame,source_history,encoder_cache,encoder_fp=frame_for_stage(stage)
    truth=labels_from_gold(gold,frame)
    text_assignments=oof_assignments(frame,'unseen_text',3,SEED+803)
    text_fold=dict(zip(frame.query_norm,text_assignments))
    miners=independent_miners(stage)
    miner=joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib') if stage=='evaluation' else joblib.load(
        ROOT/json.loads((ROOT/'artifacts/query-encoder-candidate/manifest.json').read_text())['final_ranker'])
    paths=[];audits=[]
    started=time.perf_counter()
    for mode in ['unseen_text','held_context']:
        assignments=oof_assignments(frame,mode,3,SEED+803)
        for fold in range(3):
            positions=np.flatnonzero(assignments==fold)
            subset=frame.iloc[positions]
            # Exclusions are identical to the cached corresponding query encoder.
            history,_,_=history_for_queries(source_history,subset,[truth[i] for i in positions],
                ITEM_IDS,mode,.9,SEED+804+fold)
            bundle=joblib.load(encoder_cache/f'{mode}_{fold}_queries_{encoder_fp}.joblib')
            assert bundle['history_sha256']==history_digest(history)
            LEARNED_LOOKUP=dict(zip(bundle['texts'],bundle['vectors']))
            audit=history_coverage(subset,labels_from_gold(gold,subset),history,ITEM_IDS)
            assert audit['own_context_overlap']==0
            audit.update(mode=mode,fold=fold);audits.append(audit)
            QUALITY_BANK=QualityEvidence(history)
            neighbors=evidence(history,subset)
            retriever_history=V5History(history)
            del history,bundle;gc.collect()
            for start in range(0,len(subset),V10_CONFIG['chunk']):
                chunk=subset.iloc[start:start+V10_CONFIG['chunk']]
                path=V10_CACHE/f'{stage}_{mode}_{fold}_{start}_{V10_FP}.joblib'
                paths.append(path)
                if path.exists():continue
                records=retrieve_context_features(chunk,retriever_history,progress_every=100000)
                original=[v5_features(q,r,retriever_history) for q,r in zip(chunk.itertuples(index=False),records)]
                base=matrices(chunk,records,original,tuple(n[start:start+len(chunk)] for n in neighbors))
                features=extra_matrices(chunk,records,base)
                saved=[]
                for q,(ids,raw),x in zip(chunk.itertuples(index=False),records,features):
                    target=np.isin(ids,list(gold[q.context_key]))
                    pos=np.flatnonzero(target);neg=np.flatnonzero(~target)
                    if not len(pos) or not len(neg):continue
                    ordinary=miner.predict(x[neg,:51],num_iteration=400)
                    independent=miners[text_fold[q.query_norm]].predict(x[neg][:,MINER_COLUMNS],num_iteration=200)
                    hard=np.unique(np.concatenate([
                        neg[stable_topk(ordinary,500)],neg[stable_topk(independent,120)],
                        neg[stable_topk(raw[neg,7],160)],neg[stable_topk(x[neg,45],160)]]))
                    tail=np.setdiff1d(neg,hard)
                    seed=int(hashlib.sha256(f'{stage}:{mode}:{q.context_key}'.encode()).hexdigest()[:8],16)
                    rng=np.random.default_rng(seed)
                    chosen=np.sort(np.concatenate([pos,hard,rng.choice(tail,min(len(tail),96),replace=False)]))
                    saved.append((x[chosen],target[chosen].astype(np.uint8),q.query_norm in retriever_history.text_to_row,
                        q.context_key,q.query_norm,ids[chosen]))
                save_cache(saved,path)
                print('Quality chunk',stage,mode,fold,start,'groups',len(saved),'seconds',round(time.perf_counter()-started),flush=True)
                del records,original,base,features,saved;gc.collect()
            del retriever_history,neighbors;gc.collect()
    # Two passes avoid retaining both all chunks and their concatenation in RAM.
    sizes=[];known=[];keys=[];texts=[]
    for path in paths:
        for _,y,k,key,text,_ in joblib.load(path):
            sizes.append(len(y));known.append(k);keys.append(key);texts.append(text)
    rows=sum(sizes)
    x_path=V10_CACHE/f'{stage}_X_{V10_FP}.npy';y_path=V10_CACHE/f'{stage}_y_{V10_FP}.npy'
    x=np.lib.format.open_memmap(x_path,mode='w+',dtype=np.float32,shape=(rows,len(NEW_FEATURE_NAMES)))
    y=np.lib.format.open_memmap(y_path,mode='w+',dtype=np.uint8,shape=(rows,))
    offset=0
    for path in paths:
        for features,target,*_ in joblib.load(path):
            stop=offset+len(target);x[offset:stop]=features;y[offset:stop]=target;offset=stop
    x.flush();y.flush();del x,y;gc.collect()
    report={'stage':stage,'contexts':len(frame),'groups':len(sizes),'rows':rows,
        'mean_group_size':float(np.mean(sizes)),'sizes':sizes,'known':known,
        'context_keys':keys,'query_texts':texts,'history_audits':audits,
        'feature_names':NEW_FEATURE_NAMES,'x_file':x_path.name,'y_file':y_path.name,
        'fingerprint':V10_FP,'elapsed_seconds':round(time.perf_counter()-started,2),
        'mining':'Frozen in-sample v7 plus text-held unsupervised-column miner',
        'source_sha256':sha256_file(V10_TRAIN_DRIVER)}
    manifest_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Quality training ready',stage,rows,len(sizes),flush=True)
    return load_quality_training(stage)


def load_quality_training(stage):
    report=json.loads((V10_CACHE/f'{stage}_training_{V10_FP}.json').read_text())
    known=np.asarray(report['known'],bool);observed=known.mean()
    weights=np.where(known,known_target/max(observed,1e-6),(1-known_target)/max(1-observed,1e-6)).astype(np.float32)
    return {'X':np.load(V10_CACHE/report['x_file'],mmap_mode='r'),
        'y':np.load(V10_CACHE/report['y_file'],mmap_mode='r'),'sizes':np.asarray(report['sizes'],np.int32),
        'group_weight':weights,'report':report}


MODEL_RECIPES={
    'coverage':{'columns':list(range(70)),'trees':500,'leaves':31,'min_child':80},
    'quality':{'columns':list(range(82)),'trees':500,'leaves':31,'min_child':80},
    'quality_deeper':{'columns':list(range(82)),'trees':600,'leaves':63,'min_child':150}}


def fit_quality(data,stage,names):
    models={}
    for name in names:
        path=V10_CACHE/f'{stage}_{name}_{V10_FP}.joblib'
        if path.exists():models[name]=joblib.load(path);continue
        recipe=MODEL_RECIPES[name]
        # Advanced indexing followed by ascontiguousarray can allocate TWO huge
        # copies. Materialize a contiguous subset on disk in bounded chunks.
        if recipe['columns']==list(range(data['X'].shape[1])):
            x=data['X']
        else:
            matrix_path=V10_CACHE/f'{stage}_{name}_fit_matrix_{V10_FP}.npy'
            if not matrix_path.exists():
                target=np.lib.format.open_memmap(matrix_path,mode='w+',dtype=np.float32,
                    shape=(len(data['y']),len(recipe['columns'])))
                for start in range(0,len(target),32768):
                    target[start:start+32768]=data['X'][start:start+32768,recipe['columns']]
                target.flush();del target;gc.collect()
            x=np.load(matrix_path,mmap_mode='r')
        model=lgb.LGBMRanker(n_estimators=recipe['trees'],num_leaves=recipe['leaves'],learning_rate=.04,
            reg_lambda=15,max_bin=127,min_child_samples=recipe['min_child'],random_state=SEED,
            n_jobs=8,verbosity=-1,deterministic=True,force_col_wise=True,
            lambdarank_truncation_level=55,label_gain=[0,1])
        start=time.perf_counter();print('Fit quality',stage,name,x.shape,flush=True)
        model.fit(x,data['y'],group=data['sizes'],sample_weight=np.repeat(data['group_weight'],data['sizes']))
        save_cache(model,path);models[name]=model
        report={'recipe':recipe,'seconds':round(time.perf_counter()-start,2),'model_sha256':sha256_file(path)}
        path.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        del x;gc.collect()
    return models


def evaluate_quality(stage,mode,models,category_strength):
    global QUALITY_BANK
    frame,records,features,reference,known=v9_pool(stage,mode)
    reference=compatibility_boost(frame,records,reference,category_strength)
    truth=labels_from_gold(gold,frame)
    history=evaluation_history(frame,mode,control if stage=='development' else development)
    QUALITY_BANK=QualityEvidence(history);del history;gc.collect()
    augmented=extra_matrices(frame,records,features)
    values={('v9_category',0.):per_query_recall(top50(records,reference),truth)}
    for name,model in models.items():
        recipe=MODEL_RECIPES[name]
        prediction=predict_scores(model,[x[:,recipe['columns']] for x in augmented],recipe['trees'])
        for weight in V10_CONFIG['selection_grid'][1:]:
            scores=blend_scores(prediction,reference,weight)
            values[(name,weight)]=per_query_recall(top50(records,scores),truth)
        del prediction;gc.collect()
    del records,features,augmented,reference;gc.collect()
    return values,known


def compare_quality():
    global RESOURCES,CONTEXT_VECTORS
    install_retriever();RESOURCES=prepare_resources()
    CONTEXT_VECTORS,_=contextual_vectors()
    data=prepare_quality_training('evaluation')
    models=fit_quality(data,'evaluation',list(MODEL_RECIPES))
    del data;gc.collect()
    category=json.loads((V10_CACHE/'category_selection.json').read_text())['winner']['category_boost']
    a,_=evaluate_quality('development','unseen_text',models,category)
    b,known=evaluate_quality('development','held_context',models,category)
    rows=[dict(variant=name,weight=weight,**matched_metrics(a[(name,weight)],b[(name,weight)],known))
        for name,weight in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(V10_CACHE/'development.csv',index=False,lineterminator='\n')
    baseline=next(r for r in rows if r['variant']=='v9_category')
    eligible=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)
        &(table.held_macro>=baseline['held_macro']-.0005)]
    winner=eligible.iloc[0].to_dict()
    selection={'fingerprint':V10_FP,'winner':winner,'baseline':baseline,
        'category_boost':category,'control_used_for_selection':False,
        'recipes':MODEL_RECIPES,'config':V10_CONFIG,
        'limitations':['Reused development/control; inherited auxiliary-prior limitation.',
            'Unchosen candidates are unlabeled, not proven negative.',
            'Independent miner isolates labels by text; historical retrieval pools are not nested OOF.']}
    (V10_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('Quality selection',table.to_string(index=False),flush=True)
    chosen={winner['variant']:models[winner['variant']]} if winner['variant'] in models else {}
    c,_=evaluate_quality('control','unseen_text',chosen,category)
    score=float(c[(winner['variant'],winner['weight'])].mean())
    report={'selected_recall50':score,'v9_recall50':.95,'used_for_selection':False}
    (V10_CACHE/'control.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Quality control',json.dumps(report),flush=True)


if __name__=='__main__':
    compare_quality()
