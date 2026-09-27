"""OOF E5 scores for LambdaRank, preserving the frozen document encoder.

Six query encoders follow the exact v5 history exclusions. Ranker sampling and
labels remain unchanged; full-pool ranks are computed before negative sampling.
Development chooses the configuration; control is reported only afterwards.
"""
from pathlib import Path
import ast
import hashlib
import json
import time
import gc

QR_DRIVER=Path(__file__).resolve()
driver=QR_DRIVER.with_name('query_encoder_pilot.py')
tree=ast.parse(driver.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(driver)
exec(compile(tree,str(driver),'exec'),globals())
__file__=str(QR_DRIVER)
QR_CACHE=ROOT/'artifacts/query-encoder-rank'
QR_CACHE.mkdir(exist_ok=True)
QR_FP=hashlib.sha256((QE_FP+hashlib.sha256(QR_DRIVER.read_bytes()).hexdigest()).encode()).hexdigest()[:16]
QR_FEATURE_NAMES=['learned_cosine','learned_geo','learned_cosine_delta',
    'learned_geo_delta','learned_log_cosine_rank','learned_log_geo_rank']


def query_features(vector,ids,base):
    cosine=(semantic_index.items[ids].astype(np.float64)@vector.astype(np.float64)).astype(np.float32)
    geographic=semantic_affinity(cosine,base[:,4])
    return np.column_stack([cosine,geographic,cosine-base[:,6],geographic-base[:,7],
        np.log1p(rankdata(-cosine,method='min')),np.log1p(rankdata(-geographic,method='min'))]).astype(np.float32)


def fit_fold_encoder(history,frame,tag,document_ids,documents,space):
    path=QR_CACHE/f'{tag}_queries_{QR_FP}.joblib'
    digest=history_digest(history)
    if USE_CACHE and path.exists():
        cached=joblib.load(path)
        assert cached['history_sha256']==digest and cached['texts']==sorted(set(frame.query_norm))
        return cached['vectors'],cached['audit']
    positive_map=history.groupby('query_norm',sort=True).item_id.agg(lambda s:sorted(set(s))).to_dict()
    texts=sorted(positive_map)
    assert not set(frame.context_key)&set(history.context_key)
    item_rows={item:i for i,item in enumerate(document_ids)}
    assert set(history.item_id).issubset(item_rows)
    teacher_lookup={}
    for row,text in enumerate(space['inputs']['query']):teacher_lookup.setdefault(text,row)
    teacher=np.stack([space['dense'][teacher_lookup[text],:384] for text in texts])
    torch.manual_seed(QE_CONFIG['seed'])
    random=np.random.default_rng(QE_CONFIG['seed'])
    model=AutoModel.from_pretrained(MODEL_DIR,local_files_only=True,attn_implementation='eager').to(DEVICE)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=QE_CONFIG['learning_rate'],weight_decay=QE_CONFIG['weight_decay'])
    fixed_documents=torch.tensor(documents,device=DEVICE)
    fixed_teacher=torch.tensor(teacher,device=DEVICE)
    positive_sets={text:set(ids) for text,ids in positive_map.items()}
    started=time.perf_counter();logs=[]
    for epoch in range(QE_CONFIG['epochs']):
        model.train();order=random.permutation(len(texts));losses=[]
        for start in range(0,len(order),QE_CONFIG['batch_size']):
            rows=order[start:start+QE_CONFIG['batch_size']]
            batch_text=[texts[i] for i in rows]
            batch_ids=[positive_map[text][int(random.integers(len(positive_map[text])))] for text in batch_text]
            doc_rows=torch.tensor([item_rows[item] for item in batch_ids],device=DEVICE)
            target=torch.tensor(in_batch_targets(batch_text,batch_ids,positive_sets),device=DEVICE)
            batch=tokenizer(['query: '+text for text in batch_text],padding=True,truncation=True,
                max_length=QE_CONFIG['max_query_length'],return_tensors='pt')
            batch={key:value.to(DEVICE) for key,value in batch.items()}
            with torch.autocast(device_type=DEVICE,dtype=torch.bfloat16,enabled=DEVICE=='cuda'):
                hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
                vector=torch.nn.functional.normalize(pooled.float(),p=2,dim=1)
            logits=vector@fixed_documents[doc_rows].T/QE_CONFIG['temperature']
            contrastive=-(target*torch.nn.functional.log_softmax(logits,dim=1)).sum(dim=1).mean()
            retention=(1-(vector*fixed_teacher[rows]).sum(dim=1)).mean()
            loss=contrastive+QE_CONFIG['teacher_weight']*retention
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({'epoch':epoch+1,'mean_loss':float(np.mean(losses))})
        print('OOF query encoder',tag,'epoch',epoch+1,'loss',round(logs[-1]['mean_loss'],4),
            'seconds',round(time.perf_counter()-started,1),flush=True)
    query_texts=sorted(set(frame.query_norm))
    vectors=encode_learned(model,tokenizer,query_texts)
    audit={'tag':tag,'history_sha256':digest,'pairs':len(history),'texts':len(texts),
        'own_context_overlap':0,'logs':logs,'document_encoder_frozen':True}
    save_cache({'texts':query_texts,'vectors':vectors,'history_sha256':digest,'audit':audit},path)
    del model,optimizer,fixed_documents,fixed_teacher,teacher
    gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()
    return vectors,audit


def prepare_query_oof():
    data=joblib.load(V5_CACHE/f'evaluation_mined_{V5_FP}.joblib')
    path=QR_CACHE/f'oof_features_{QR_FP}.joblib'
    if USE_CACHE and path.exists():return data,joblib.load(path)
    document_ids,documents=training_documents()
    space=prepare_micro_space()
    extra=np.empty((len(data['y']),len(QR_FEATURE_NAMES)),np.float32)
    miner=joblib.load(V5_CACHE/f'mixed_wide_{V5_FP}.joblib')
    random=np.random.default_rng(SEED+811)
    truths=labels_from_gold(gold,training);cursor=group=0;audits=[]
    for mode in ['unseen_text','held_context']:
        assignments=oof_assignments(training,mode,3,SEED+803)
        for fold in range(3):
            positions=np.flatnonzero(assignments==fold);frame=training.iloc[positions]
            truth=[truths[i] for i in positions]
            fit,_,_=history_for_queries(training_history,frame,truth,ITEM_IDS,mode,.9,SEED+804+fold)
            vectors,audit=fit_fold_encoder(fit,frame,f'{mode}_{fold}',document_ids,documents,space)
            audit['query_text_overlap']=len(set(frame.query_norm)&set(fit.query_norm))
            if mode=='unseen_text':assert audit['query_text_overlap']==0
            audits.append(audit)
            lookup={text:i for i,text in enumerate(sorted(set(frame.query_norm)))}
            del fit
            for start in range(0,len(positions),128):
                records,matrices=joblib.load(V5_CACHE/f'evaluation_pool_{mode}_{fold}_{start}_{V5_FP}.joblib')
                for offset,((ids,base),features) in enumerate(zip(records,matrices)):
                    selected,target=sample_v5_group(ids,base,features,truth[start+offset],random,miner)
                    if not len(selected):continue
                    count=len(selected)
                    assert data['query_position'][group]==positions[start+offset] and data['mode'][group]==mode
                    assert np.array_equal(data['X'][cursor:cursor+count],features[selected],equal_nan=True)
                    assert np.array_equal(data['y'][cursor:cursor+count],target[selected])
                    vector=vectors[lookup[frame.iloc[start+offset].query_norm]]
                    extra[cursor:cursor+count]=query_features(vector,ids,base)[selected]
                    cursor+=count;group+=1
                if start%2048==0:print('OOF query features',mode,fold,min(start+128,len(frame)),flush=True)
            del vectors,records,matrices
            gc.collect()
    assert cursor==len(data['y']) and group==len(data['sizes'])
    save_cache(extra,path)
    (QR_CACHE/'oof_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    return data,extra


def evaluation_query_features(frame,stage):
    path=QR_CACHE/f'{stage}_queries_{QR_FP}.joblib'
    texts=sorted(set(frame.query_norm))
    if USE_CACHE and path.exists():
        saved=joblib.load(path);assert saved['texts']==texts
        return saved['vectors']
    if stage=='development':
        vectors=np.load(QE_CACHE/f'queries_epoch_3_{QE_FP}.npy',allow_pickle=False)
    else:
        vectors=np.load(QE_CACHE/f'control_queries_epoch_3_{QE_FP}.npy',allow_pickle=False)
    save_cache({'texts':texts,'vectors':vectors},path)
    return vectors


def run_query_rank():
    data,extra=prepare_query_oof()
    # Fixed comparison: only six learned features differ from v5.
    path=QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib'
    if USE_CACHE and path.exists():model=joblib.load(path)
    else:
        x=np.column_stack([data['X'],extra])
        model=lgb.LGBMRanker(n_estimators=400,num_leaves=31,learning_rate=.05,max_bin=127,
            min_child_samples=50,reg_lambda=10,random_state=SEED,n_jobs=8,verbosity=-1,
            deterministic=True,force_col_wise=True,lambdarank_truncation_level=55,label_gain=[0,1])
        print('Fit learned query ranker',x.shape,flush=True)
        model.fit(x,data['y'],group=data['sizes'],sample_weight=np.repeat(data['group_weight'],data['sizes']))
        save_cache(model,path);del x
    del data,extra;gc.collect()
    vectors=evaluation_query_features(development,'development')
    lookup={text:i for i,text in enumerate(sorted(set(development.query_norm)))}
    bundles={}
    for mode in ['unseen_text','held_context']:
        records,base,known,audit=evaluation_features(development,mode,'development',control)
        matrix=[np.column_stack([features,query_features(vectors[lookup[query]],ids,raw)])
            for query,(ids,raw),features in zip(development.query_norm,records,base)]
        baseline=joblib.load(MICRO_CACHE/f'development_{mode}_v5scores_{MICRO_FP}.joblib')
        bundles[mode]=(records,matrix,known,baseline)
    a=bundles['unseen_text'];b=bundles['held_context']
    baseline=development_metrics(top50(a[0],a[3]),top50(b[0],b[3]),b[2])
    rows=[{'variant':'v5','trees':400,'weight':0.,**baseline}]
    # Include the already fixed pilot as a reference, not an extra control search.
    pilot=json.loads((QE_CACHE/'pilot_report.json').read_text())['winner']
    rows.append({'variant':'direct_query_blend','trees':3,'weight':pilot['weight'],
        **{key:pilot[key] for key in baseline}})
    for trees in [200,400]:
        first=predict_scores(model,a[1],trees);second=predict_scores(model,b[1],trees)
        for weight in [.5,.75,1.]:
            metrics=development_metrics(top50(a[0],blend_scores(first,a[3],weight)),
                top50(b[0],blend_scores(second,b[3],weight)),b[2])
            rows.append({'variant':'query_ranker','trees':trees,'weight':weight,**metrics})
    table=pd.DataFrame(rows).sort_values(['matched_recall50','variant','trees','weight'],ascending=[False,True,True,True])
    table.to_csv(QR_CACHE/'ablation.csv',index=False,lineterminator='\n')
    winner=table[table.unseen_macro_recall50>=baseline['unseen_macro_recall50']-.002].iloc[0].to_dict()
    selection={'winner':winner,'baseline':baseline,'fingerprint':QR_FP,'control_used_for_selection':False,
        'feature_names':QR_FEATURE_NAMES,'limitation':'Evaluation encoder excludes all dev/control query texts; held-mode features are conservative.'}
    (QR_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print('QUERY RANK SELECTED',json.dumps(selection),flush=True)
    del bundles,a,b;gc.collect()
    records,base,known,audit=evaluation_features(control,'unseen_text','control',development)
    original=baseline_v5(records,base)
    vectors=evaluation_query_features(control,'control')
    lookup={text:i for i,text in enumerate(sorted(set(control.query_norm)))}
    added=[query_features(vectors[lookup[text]],ids,raw) for text,(ids,raw) in zip(control.query_norm,records)]
    if winner['variant']=='v5':chosen=original
    elif winner['variant']=='direct_query_blend':chosen=blend_scores([x[:,1] for x in added],original,float(winner['weight']))
    else:
        chosen=blend_scores(predict_scores(model,[np.column_stack([x,y]) for x,y in zip(base,added)],int(winner['trees'])),original,float(winner['weight']))
    truth=labels_from_gold(gold,control)
    before=per_query_recall(top50(records,original),truth);after=per_query_recall(top50(records,chosen),truth)
    delta=after-before;random=np.random.default_rng(SEED+908)
    boot=[float(random.choice(delta,len(delta),replace=True).mean()) for _ in range(4000)]
    result={'v5_recall50':float(before.mean()),'selected_recall50':float(after.mean()),
        'delta':float(delta.mean()),'improved':int((delta>0).sum()),'worsened':int((delta<0).sum()),
        'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'used_for_selection':False,
        'history_audit':audit,'answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation':'Previously viewed control; frozen v4 auxiliary-history limitation.'}
    assert result['answer_unchanged']
    (QR_CACHE/'control.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('QUERY RANK CONTROL',json.dumps(result),flush=True)


if __name__=='__main__':
    run_query_rank()
