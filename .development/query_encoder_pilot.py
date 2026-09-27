"""Train only E5 query encoder; fixed document vectors and isolated dev labels.

Contrastive batches contain unique query texts and all known in-batch positives.
The full permitted history supplies labels, including items outside the benchmark.
A small frozen-teacher penalty limits drift. This bounded pilot never writes the
main submission and does not fit a downstream ranker on in-sample encoder scores.
"""
from pathlib import Path
import ast
import hashlib
import json
import time
import gc

QUERY_DRIVER=Path(__file__).resolve()
micro_source=QUERY_DRIVER.with_name('microcat_v6.py')
tree=ast.parse(micro_source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(micro_source)
exec(compile(tree,str(micro_source),'exec'),globals())
__file__=str(QUERY_DRIVER)
QE_CACHE=ROOT/'artifacts/query-encoder-pilot'
QE_CACHE.mkdir(exist_ok=True)
QE_CONFIG={'epochs':3,'checkpoints':[1,3],'batch_size':128,'learning_rate':2e-5,
           'weight_decay':.01,'temperature':.05,'teacher_weight':.2,'max_query_length':64,
           'seed':SEED+905,'document_encoder_frozen':True,'negative_source':'in-batch',
           'multiple_positive_mask':True,'filters_in_encoder':False}
QE_FP=hashlib.sha256(json.dumps({'config':QE_CONFIG,'inputs':input_hashes,'v5':V5_FP,
    'model':model_manifest,'source':hashlib.sha256(QUERY_DRIVER.read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()[:16]


def in_batch_targets(query_texts,document_ids,positive_map):
    targets=np.array([[item in positive_map[query] for item in document_ids] for query in query_texts],np.float32)
    assert (targets.sum(axis=1)>0).all()
    return targets/targets.sum(axis=1,keepdims=True)


def training_documents():
    ids=sorted(training_history.item_id.unique())
    path=QE_CACHE/f'document_vectors_{QE_FP}.npy'
    if USE_CACHE and path.exists():
        vectors=np.load(path,allow_pickle=False)
    else:
        vectors=np.empty((len(ids),384),np.float32)
        missing=[];positions=[]
        for row,item_id in enumerate(ids):
            if item_id in ITEM_TO_ROW:
                vectors[row]=semantic_index.items[ITEM_TO_ROW[item_id]]
            else:
                missing.append(item_id);positions.append(row)
        # v5 initialization frees the large raw train frame; reload only text columns.
        metadata=pd.read_parquet(ROOT/'train.parquet',columns=['item_id','item_title_raw','item_description_raw','item_infm_params_text'])
        metadata=metadata.drop_duplicates('item_id').set_index('item_id')
        documents=metadata.loc[missing].fillna('')
        text=(documents.item_title_raw+'. '+documents.item_infm_params_text.str.slice(0,SEMANTIC_CONFIG['params_chars'])+
              '. '+documents.item_description_raw.str.slice(0,SEMANTIC_CONFIG['description_chars'])).tolist()
        vectors[positions]=semantic_index.encode(text,'passage: ')
        save_array(vectors,path)
        semantic_index.encoder=semantic_index.tokenizer=None
        del metadata,documents,text
        gc.collect()
        if DEVICE=='cuda':torch.cuda.empty_cache()
    assert vectors.shape==(len(ids),384) and np.isfinite(vectors).all()
    return ids,vectors


def encode_learned(model,tokenizer,texts):
    order=np.argsort([len(text) for text in texts],kind='stable')
    output=np.empty((len(texts),384),np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(order),128):
            positions=order[start:start+128]
            batch=tokenizer(['query: '+texts[i] for i in positions],padding=True,truncation=True,
                max_length=QE_CONFIG['max_query_length'],return_tensors='pt')
            batch={key:value.to(DEVICE) for key,value in batch.items()}
            hidden=model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
            output[positions]=torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy()
    assert np.isfinite(output).all() and np.allclose(np.linalg.norm(output,axis=1),1,atol=2e-5)
    return output


def train_query_encoder():
    ids,documents=training_documents()
    item_rows={item:i for i,item in enumerate(ids)}
    positives=training_history.groupby('query_norm',sort=True).item_id.agg(lambda values:sorted(set(values))).to_dict()
    texts=sorted(positives)
    assert not set(texts)&set(all_evaluation.query_norm)
    evaluation_texts=sorted(set(development.query_norm))
    last=QE_CACHE/f'epoch_3_{QE_FP}'
    if USE_CACHE and (last/'model.safetensors').exists():
        print('Loaded completed query-encoder pilot checkpoints.',flush=True)
        return
    space=prepare_micro_space()
    lookup={}
    for row,query in enumerate(space['inputs']['query']):
        lookup.setdefault(query,row)
    teacher=np.stack([space['dense'][lookup[query],:384] for query in texts])
    del space,lookup
    gc.collect()
    torch.manual_seed(QE_CONFIG['seed'])
    model=AutoModel.from_pretrained(MODEL_DIR,local_files_only=True,attn_implementation='eager').to(DEVICE)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,use_fast=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=QE_CONFIG['learning_rate'],weight_decay=QE_CONFIG['weight_decay'])
    fixed_documents=torch.tensor(documents,device=DEVICE)
    fixed_teacher=torch.tensor(teacher,device=DEVICE)
    assert not fixed_documents.requires_grad and not fixed_teacher.requires_grad
    positive_sets={query:set(values) for query,values in positives.items()}
    random=np.random.default_rng(QE_CONFIG['seed'])
    logs=[];started=time.perf_counter()
    if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,QE_CONFIG['epochs']+1):
        model.train()
        order=random.permutation(len(texts))
        losses=[]
        for start in range(0,len(order),QE_CONFIG['batch_size']):
            rows=order[start:start+QE_CONFIG['batch_size']]
            batch_text=[texts[i] for i in rows]
            document_ids=[positives[query][int(random.integers(len(positives[query])))] for query in batch_text]
            document_rows=torch.tensor([item_rows[item] for item in document_ids],device=DEVICE)
            target=torch.tensor(in_batch_targets(batch_text,document_ids,positive_sets),device=DEVICE)
            batch=tokenizer(['query: '+query for query in batch_text],padding=True,truncation=True,
                max_length=QE_CONFIG['max_query_length'],return_tensors='pt')
            batch={key:value.to(DEVICE) for key,value in batch.items()}
            with torch.autocast(device_type=DEVICE,dtype=torch.bfloat16,enabled=DEVICE=='cuda'):
                hidden=model(**batch).last_hidden_state
                mask=batch['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
                query_vectors=torch.nn.functional.normalize(pooled.float(),p=2,dim=1)
            logits=query_vectors@fixed_documents[document_rows].T/QE_CONFIG['temperature']
            contrastive=-(target*torch.nn.functional.log_softmax(logits,dim=1)).sum(dim=1).mean()
            retention=(1-(query_vectors*fixed_teacher[rows]).sum(dim=1)).mean()
            loss=contrastive+QE_CONFIG['teacher_weight']*retention
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            if start%(128*100)==0:
                print('Query encoder epoch',epoch,'queries',min(start+len(rows),len(texts)),'/',len(texts),
                    'loss',round(float(np.mean(losses[-100:])),4),'seconds',round(time.perf_counter()-started,1),flush=True)
        if epoch in QE_CONFIG['checkpoints']:
            path=QE_CACHE/f'epoch_{epoch}_{QE_FP}'
            path.mkdir(exist_ok=True)
            model.save_pretrained(path,safe_serialization=True)
            learned=encode_learned(model,tokenizer,evaluation_texts)
            save_array(learned,QE_CACHE/f'queries_epoch_{epoch}_{QE_FP}.npy')
        logs.append({'epoch':epoch,'mean_loss':float(np.mean(losses)),
                     'elapsed_seconds':round(time.perf_counter()-started,2)})
    report={'fingerprint':QE_FP,'config':QE_CONFIG,'training_pairs_available':len(training_history),
        'training_query_texts':len(texts),'document_vectors':len(ids),'sampled_pairs':len(texts)*QE_CONFIG['epochs'],
        'development_text_overlap':0,'document_vectors_sha256':sha256_file(QE_CACHE/f'document_vectors_{QE_FP}.npy'),
        'original_encoder_weights_sha256':sha256_file(MODEL_DIR/'model.safetensors'),
        'gpu_peak_allocated_MiB':round(torch.cuda.max_memory_allocated()/2**20,1) if DEVICE=='cuda' else None,'logs':logs}
    (QE_CACHE/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    del model,optimizer,fixed_documents,fixed_teacher,documents,teacher
    gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()
    print('QUERY TRAINED',json.dumps(report),flush=True)


def evaluate_query_encoder():
    lookup={text:i for i,text in enumerate(sorted(set(development.query_norm)))}
    learned={epoch:np.load(QE_CACHE/f'queries_epoch_{epoch}_{QE_FP}.npy',allow_pickle=False) for epoch in QE_CONFIG['checkpoints']}
    bundles={}
    for mode in ['unseen_text','held_context']:
        records,features,known,audit=evaluation_features(development,mode,'development',control)
        baseline=joblib.load(MICRO_CACHE/f'development_{mode}_v5scores_{MICRO_FP}.joblib')
        scores={'frozen':([base[:,6] for _,base in records],[base[:,7] for _,base in records])}
        for epoch,vectors in learned.items():
            cosine=[];geographic=[]
            for query,(ids,base) in zip(development.query_norm,records):
                values=(semantic_index.items[ids].astype(np.float64)@vectors[lookup[query]].astype(np.float64)).astype(np.float32)
                cosine.append(values);geographic.append(semantic_affinity(values,base[:,4]))
            scores[f'epoch_{epoch}']=(cosine,geographic)
        bundles[mode]=(records,features,known,audit,baseline,scores)
    rows=[]
    for variant in ['frozen','epoch_1','epoch_3']:
        for affinity in ['cosine','geo']:
            index=0 if affinity=='cosine' else 1
            for weight in [0.,.02,.05,.1,.2]:
                a=bundles['unseen_text'];b=bundles['held_context']
                metrics=development_metrics(top50(a[0],blend_scores(a[5][variant][index],a[4],weight)),
                    top50(b[0],blend_scores(b[5][variant][index],b[4],weight)),b[2])
                rows.append({'variant':variant,'affinity':affinity,'weight':weight,**metrics})
    table=pd.DataFrame(rows).sort_values(['matched_recall50','variant','affinity','weight'],ascending=[False,True,True,True])
    table.to_csv(QE_CACHE/'pilot_ablation.csv',index=False,lineterminator='\n')
    baseline=table[table.weight==0].iloc[0].to_dict()
    winner=table.iloc[0].to_dict()
    frozen=table[table.variant=='frozen'].iloc[0].to_dict()
    result={'pilot_only':True,'fingerprint':QE_FP,'baseline':baseline,'winner':winner,
        'best_frozen_score_blend':frozen,'delta_vs_v5':winner['matched_recall50']-baseline['matched_recall50'],
        'control_evaluated':False,'answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation':'Direct score blends on previously-viewed full development; no downstream ranker trained on encoder scores.',
        'next':'If useful beyond the frozen blend, produce OOF encoder features and refit the selector.'}
    assert result['answer_unchanged']
    (QE_CACHE/'pilot_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('QUERY PILOT',json.dumps(result),flush=True)


if __name__=='__main__':
    train_query_encoder()
    evaluate_query_encoder()
