"""Frozen multilingual cross-encoder pilot on 300/500 candidates, offline.

Uses only development: 600 unseen-text and 400 actually-known held contexts.
Never changes answer.csv or opens held control. The chosen score combination
is a pilot, not a final trained selector or a claim about hidden benchmark quality.
"""
from pathlib import Path
import ast
import json
import hashlib
import time
import gc

PILOT_DRIVER=Path(__file__).resolve()
micro_source=PILOT_DRIVER.with_name('microcat_v6.py')
tree=ast.parse(micro_source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(micro_source)
exec(compile(tree,str(micro_source),'exec'),globals())
__file__=str(PILOT_DRIVER)
from transformers import AutoModelForSequenceClassification
CE_DIR=ROOT/'models/mmarco-miniLM-cross-encoder'
CE_MANIFEST=json.loads((CE_DIR/'manifest.json').read_text())
for name,expected in CE_MANIFEST['sha256'].items():
    assert sha256_file(CE_DIR/name)==expected
CE_CACHE=ROOT/'artifacts/cross-encoder-pilot'
CE_CACHE.mkdir(exist_ok=True)
CE_CONFIG={'max_length':192,'batch_size':64,'seed':SEED+903,'unseen_contexts':600,
           'known_contexts':400,'params_chars':256,'description_chars':1200,'filter_chars':120,
           'shortlists':[300,500],'dtype':'float16' if DEVICE=='cuda' else 'float32'}
CE_FP=hashlib.sha256(json.dumps({'model':CE_MANIFEST,'config':CE_CONFIG,'v5':V5_FP,
    'source':hashlib.sha256(PILOT_DRIVER.read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()[:16]


def ce_scores(model,tokenizer,frame,records,base_scores):
    selected=[stable_topk(score,500) for score in base_scores]
    candidate_ids=np.concatenate([record[0][position] for record,position in zip(records,selected)])
    query_rows=np.repeat(np.arange(len(frame)),[len(x) for x in selected])
    query_texts=(frame.query_norm+'. '+frame.search_infm_params_text.str.slice(0,CE_CONFIG['filter_chars'])).tolist()
    documents=(items.item_title_raw+'. '+items.item_infm_params_text.str.slice(0,CE_CONFIG['params_chars'])+'. '+
               items.item_description_raw.str.slice(0,CE_CONFIG['description_chars'])).tolist()
    lengths=np.array([len(documents[i])+len(query_texts[q]) for i,q in zip(candidate_ids,query_rows)])
    order=np.argsort(lengths,kind='stable')
    output=np.empty(len(order),np.float32)
    started=time.perf_counter()
    if DEVICE=='cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0,len(order),CE_CONFIG['batch_size']):
            positions=order[start:start+CE_CONFIG['batch_size']]
            batch=tokenizer([query_texts[query_rows[i]] for i in positions],
                [documents[candidate_ids[i]] for i in positions],padding=True,truncation=True,
                max_length=CE_CONFIG['max_length'],return_tensors='pt')
            batch={key:value.to(DEVICE) for key,value in batch.items()}
            output[positions]=model(**batch).logits[:,0].float().cpu().numpy()
            if start%32000==0:
                print('Cross-encoder pairs',min(start+len(positions),len(order)),'/',len(order),
                      'seconds',round(time.perf_counter()-started,1),flush=True)
    assert np.isfinite(output).all()
    split=np.split(output,np.cumsum([len(x) for x in selected])[:-1])
    performance={'pairs':len(output),'seconds':round(time.perf_counter()-started,2),
        'pairs_per_second':round(len(output)/max(time.perf_counter()-started,.01),1),
        'gpu_peak_allocated_MiB':round(torch.cuda.max_memory_allocated()/2**20,1) if DEVICE=='cuda' else None}
    return selected,split,performance


def run_pilot():
    tokenizer=AutoTokenizer.from_pretrained(CE_DIR,local_files_only=True,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(CE_DIR,local_files_only=True,
        attn_implementation='eager',torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32).to(DEVICE).eval()
    bundles={}
    random=np.random.default_rng(CE_CONFIG['seed'])
    for mode in ['unseen_text','held_context']:
        path=CE_CACHE/f'{mode}_{CE_FP}.joblib'
        if USE_CACHE and path.exists():
            bundles[mode]=joblib.load(path)
            continue
        records,matrices,known,audit=evaluation_features(development,mode,'development',control)
        eligible=np.arange(len(development)) if mode=='unseen_text' else np.flatnonzero(known)
        count=CE_CONFIG['unseen_contexts'] if mode=='unseen_text' else CE_CONFIG['known_contexts']
        positions=np.sort(random.choice(eligible,count,replace=False))
        frame=development.iloc[positions]
        records=[records[i] for i in positions];matrices=[matrices[i] for i in positions]
        score_file=MICRO_CACHE/f'development_{mode}_v5scores_{MICRO_FP}.joblib'
        if score_file.exists():
            cached_scores=joblib.load(score_file)
            scores=[cached_scores[i] for i in positions]
            del cached_scores
        else:
            scores=baseline_v5(records,matrices)
        selected,logits,performance=ce_scores(model,tokenizer,frame,records,scores)
        bundle={'frame':frame,'records':records,'baseline':scores,'selected':selected,'logits':logits,
                'performance':performance,'history_audit':audit,'positions':positions}
        save_cache(bundle,path)
        bundles[mode]=bundle
        print('CE performance',mode,json.dumps(performance),flush=True)
        del matrices
        gc.collect()
    rows=[]
    for shortlist in CE_CONFIG['shortlists']:
        for weight in [0.,.02,.05,.1,.2,.4]:
            metrics={}
            for mode,bundle in bundles.items():
                records=bundle['records'];scores=bundle['baseline']
                truth=labels_from_gold(gold,bundle['frame'])
                output=[]
                for base,positions,logits in zip(scores,bundle['selected'],bundle['logits']):
                    value=(1-weight)/(60+rankdata(-base,method='min'))
                    if weight:
                        selected=positions[:shortlist]
                        value[selected]+=weight/(60+rankdata(-logits[:shortlist],method='min'))
                    output.append(value)
                recalls=per_query_recall(top50(records,output),truth)
                metrics[mode+'_recall50']=float(recalls.mean())
                metrics[mode+'_shortlist_recall']=float(per_query_recall(
                    [record[0][positions[:shortlist]] for record,positions in zip(records,bundle['selected'])],truth).mean())
            metrics['mixed_recall50']=(1-known_target)*metrics['unseen_text_recall50']+known_target*metrics['held_context_recall50']
            rows.append({'shortlist':shortlist,'weight':weight,**metrics})
    table=pd.DataFrame(rows).sort_values(['mixed_recall50','shortlist','weight'],ascending=[False,True,True])
    table.to_csv(CE_CACHE/'pilot_ablation.csv',index=False,lineterminator='\n')
    winner=table.iloc[0].to_dict()
    baseline=table[table.weight==0].iloc[0].to_dict()
    result={'pilot_only':True,'fingerprint':CE_FP,'model':CE_MANIFEST,'config':CE_CONFIG,
        'winner':winner,'baseline':baseline,'delta':winner['mixed_recall50']-baseline['mixed_recall50'],
        'performance':{mode:b['performance'] for mode,b in bundles.items()},
        'control_evaluated':False,'answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation':'Small previously-viewed development subsets; direct score blend, not a trained ranker feature.',
        'next':'If useful, train a selector with the CE score and verify on the full development set.'}
    assert result['answer_unchanged']
    (CE_CACHE/'pilot_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('CE PILOT',json.dumps(result),flush=True)


if __name__=='__main__':
    run_pilot()
