"""Task-specific cross-encoder pilot, isolated from development labels.

The ready local mMARCO model is fine-tuned on a bounded text-fold training sample.
All observed positives are retained. Unchosen candidates are noisy negatives:
their loss is downweighted, and the final CE only complements the baseline.
This pilot is not silently promoted or claimed to be full OOF ranker training.
"""
from pathlib import Path
import ast
import gc
import hashlib
import json
import time

CE_DRIVER=Path(__file__).resolve()
source=CE_DRIVER.with_name('quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(CE_DRIVER)
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from quality_signals_v10 import query_filter_text

CE_CACHE=ROOT/'artifacts/cross-finetune-v10';CE_CACHE.mkdir(exist_ok=True)
CE_CONFIG={'training_text_fold':0,'max_contexts':6000,'epochs':2,'batch_size':32,
    'max_length':192,'learning_rate':5e-6,'weight_decay':.01,'hard_negatives':8,
    'random_negatives':4,'negative_weight':.25,'top_candidates':200,'seed':SEED+1901}
CE_FP=hashlib.sha256(json.dumps({'config':CE_CONFIG,'v9':V9_FP,
    'source':sha256_file(CE_DRIVER),'model':sha256_file(ROOT/'models/mmarco-miniLM-cross-encoder/model.safetensors')},
    sort_keys=True).encode()).hexdigest()[:16]


def texts_for_items():
    return (items.item_title_raw+'. '+items.item_infm_params_text.str.slice(0,500)+
        '. '+items.item_description_raw.str.slice(0,1800)).to_numpy()


def pilot_pairs():
    path=CE_CACHE/f'pairs_{CE_FP}.joblib'
    if path.exists():return joblib.load(path)
    # v10 chunks preserve context keys and selected corpus row IDs explicitly.
    ready=list(V10_CACHE.glob('evaluation_unseen_text_0_*_*.joblib'))
    fingerprints={p.stem.split('_')[-1] for p in ready}
    assert fingerprints,'Prepare v10 training chunks before the CE pilot'
    # An interrupted earlier recipe may have a few chunks; prefer the sample
    # with the greatest completed coverage and retain its fingerprint in audit.
    fp=max(fingerprints,key=lambda value:len(list(V10_CACHE.glob(f'evaluation_unseen_text_0_*_{value}.joblib'))))
    queries_by_key=training.set_index('context_key')
    examples=[];contexts=set();texts=set();rng=np.random.default_rng(CE_CONFIG['seed'])
    for path in sorted(V10_CACHE.glob(f'evaluation_unseen_text_0_*_{fp}.joblib')):
        for x,y,known,key,text,ids in joblib.load(path):
            if len(contexts)>=CE_CONFIG['max_contexts']:break
            query=queries_by_key.loc[key]
            query_text=query_filter_text(query.query_norm,query.search_infm_params_text)
            positives=np.flatnonzero(y);negatives=np.flatnonzero(~y.astype(bool))
            hard=negatives[stable_topk(x[negatives,32],CE_CONFIG['hard_negatives'])]
            tail=np.setdiff1d(negatives,hard)
            random=rng.choice(tail,min(len(tail),CE_CONFIG['random_negatives']),replace=False)
            for row in np.concatenate([positives,hard,random]):
                examples.append((query_text,int(ids[row]),float(y[row])))
            contexts.add(key);texts.add(text)
        if len(contexts)>=CE_CONFIG['max_contexts']:break
    assert len(contexts)>=5000,'Wait for text fold 0 chunks to finish before CE pilot'
    assert not texts & set(development.query_norm) and not texts & set(control.query_norm)
    bundle={'examples':examples,'contexts':len(contexts),'texts':len(texts),
        'own_development_text_overlap':0,'sampling_fingerprint':fp}
    save_cache(bundle,CE_CACHE/f'pairs_{CE_FP}.joblib')
    return bundle


def fit_pilot():
    directory=CE_CACHE/f'pilot_{CE_FP}'
    if (directory/'model.safetensors').exists():return directory
    assert DEVICE=='cuda','CE training pilot requires available local GPU'
    pairs=pilot_pairs();examples=pairs['examples'];documents=texts_for_items()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/mmarco-miniLM-cross-encoder',local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(ROOT/'models/mmarco-miniLM-cross-encoder',
        local_files_only=True,attn_implementation='eager').to(DEVICE)
    torch.manual_seed(CE_CONFIG['seed'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=CE_CONFIG['learning_rate'],weight_decay=CE_CONFIG['weight_decay'])
    rng=np.random.default_rng(CE_CONFIG['seed']);logs=[];started=time.perf_counter()
    for epoch in range(CE_CONFIG['epochs']):
        model.train();order=rng.permutation(len(examples));losses=[]
        for start in range(0,len(order),CE_CONFIG['batch_size']):
            selected=[examples[i] for i in order[start:start+CE_CONFIG['batch_size']]]
            batch=tokenizer([e[0] for e in selected],[documents[e[1]] for e in selected],
                truncation=True,padding=True,max_length=CE_CONFIG['max_length'],return_tensors='pt')
            batch={k:v.to(DEVICE) for k,v in batch.items()}
            labels=torch.tensor([e[2] for e in selected],device=DEVICE)
            weights=torch.where(labels>0,1.,CE_CONFIG['negative_weight'])
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits=model(**batch).logits.flatten().float()
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weights).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
            if start%(CE_CONFIG['batch_size']*300)==0:
                print('Fine-tune CE',epoch+1,start,len(order),'loss',round(float(np.mean(losses[-300:])),4),
                    'seconds',round(time.perf_counter()-started,1),flush=True)
        logs.append({'epoch':epoch+1,'mean_loss':float(np.mean(losses))})
    directory.mkdir(exist_ok=True);model.save_pretrained(directory,safe_serialization=True)
    report={'config':CE_CONFIG,'logs':logs,'contexts':pairs['contexts'],'pairs':len(examples),
        'seconds':round(time.perf_counter()-started,2),'development_text_overlap':0,
        'model_sha256':sha256_file(directory/'model.safetensors'),
        'limitations':['Only 6000 training contexts; noisy downweighted unchosen negatives.']}
    (CE_CACHE/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    del model,optimizer,pairs,examples,documents;gc.collect();torch.cuda.empty_cache()
    return directory


def score_pilot(stage,mode,model,tokenizer,documents):
    frame,records,features,reference,known=v9_pool(stage,mode)
    category=json.loads((V10_CACHE/'category_selection.json').read_text())['winner']['category_boost']
    reference=compatibility_boost(frame,records,reference,category)
    cache=CE_CACHE/f'{stage}_{mode}_scores_{CE_FP}.joblib'
    if cache.exists():scores=joblib.load(cache)
    else:
        scores=[]
        with torch.inference_mode():
            for row,(q,(ids,_),base) in enumerate(zip(frame.itertuples(index=False),records,reference)):
                selected=stable_topk(base,CE_CONFIG['top_candidates'])
                query=query_filter_text(q.query_norm,q.search_infm_params_text)
                result=np.full(len(ids),-1e6,np.float32)
                for start in range(0,len(selected),64):
                    positions=selected[start:start+64]
                    batch=tokenizer([query]*len(positions),documents[ids[positions]].tolist(),
                        truncation=True,padding=True,max_length=CE_CONFIG['max_length'],return_tensors='pt')
                    batch={k:v.to(DEVICE) for k,v in batch.items()}
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        result[positions]=model(**batch).logits.flatten().float().cpu().numpy()
                scores.append(result)
                if row%200==0:print('CE pilot scores',stage,mode,row,len(frame),flush=True)
        save_cache(scores,cache)
    truth=labels_from_gold(gold,frame)
    values={0.:per_query_recall(top50(records,reference),truth)}
    for weight in [.025,.05,.1]:
        values[weight]=per_query_recall(top50(records,blend_scores(scores,reference,weight)),truth)
    del records,features,scores,reference;gc.collect()
    return values,known


def compare_pilot():
    directory=fit_pilot()
    model=AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True,
        attn_implementation='eager').to(DEVICE).eval()
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/mmarco-miniLM-cross-encoder',local_files_only=True)
    documents=texts_for_items()
    a,_=score_pilot('development','unseen_text',model,tokenizer,documents)
    b,known=score_pilot('development','held_context',model,tokenizer,documents)
    rows=[dict(weight=w,**matched_metrics(a[w],b[w],known)) for w in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(CE_CACHE/'development.csv',index=False,lineterminator='\n')
    baseline=next(r for r in rows if r['weight']==0.)
    winner=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)
        &(table.held_macro>=baseline['held_macro']-.0005)].iloc[0].to_dict()
    report={'winner':winner,'baseline':baseline,'control_used_for_selection':False,
        'model':directory.relative_to(ROOT).as_posix(),'fingerprint':CE_FP,
        'limitations':['Reused development/control; fixed v9 shortlist of 200 candidates.']}
    (CE_CACHE/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CE fine-tune pilot',table.to_string(index=False),flush=True)
    if winner['weight']>0:
        c,_=score_pilot('control','unseen_text',model,tokenizer,documents)
        (CE_CACHE/'control.json').write_text(json.dumps({'baseline':float(c[0.].mean()),
            'selected':float(c[winner['weight']].mean()),'used_for_selection':False},indent=2),encoding='utf-8')


if __name__=='__main__':compare_pilot()
