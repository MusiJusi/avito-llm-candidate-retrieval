"""Bounded frozen-E5 document-field ablation; no new model or external API.

Two document representations test truncation and parameter-heavy announcements.
Their scores complement v9; geography remains in the semantic channel. The
predetermined small grid is selected on development, then checked on control.
The vectors can also become extra ranker features using v10 saved candidate IDs.
"""
from pathlib import Path
import ast
import gc
import json
import hashlib

FIELD_DRIVER=Path(__file__).resolve()
source=FIELD_DRIVER.with_name('quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(FIELD_DRIVER)
from quality_signals_v10 import query_filter_text

FIELD_CONFIGS={
    'service_fields':{'length':192,'params_chars':1800,'description_chars':0},
    'long_description':{'length':384,'params_chars':600,'description_chars':3500}}
FIELD_CACHE=ROOT/'artifacts/semantic-fields-v10';FIELD_CACHE.mkdir(exist_ok=True)


def document_fields():
    result={};manifest={}
    original=dict(SEMANTIC_CONFIG)
    for name,config in FIELD_CONFIGS.items():
        fp=hashlib.sha256(json.dumps({'config':config,'items':input_hashes['benchmark_items.parquet'],
            'model':model_manifest,'source':sha256_file(FIELD_DRIVER)},sort_keys=True).encode()).hexdigest()[:16]
        path=FIELD_CACHE/f'{name}_{fp}.npy'
        if not path.exists():
            texts=(items.item_title_raw+'. '+items.item_infm_params_text.str.slice(0,config['params_chars']))
            if config['description_chars']:
                texts=texts+'. '+items.item_description_raw.str.slice(0,config['description_chars'])
            SEMANTIC_CONFIG.update(max_length=config['length'],batch_size=32)
            print('Encode document fields',name,config,flush=True)
            vectors=semantic_index.encode(texts.tolist(),'passage: ')
            save_array(vectors,path);del vectors,texts;gc.collect()
        result[name]=np.load(path,mmap_mode='r')
        manifest[name]={'path':path.relative_to(ROOT).as_posix(),'sha256':sha256_file(path),'config':config}
    SEMANTIC_CONFIG.clear();SEMANTIC_CONFIG.update(original)
    semantic_index.encoder=semantic_index.tokenizer=None;gc.collect()
    if DEVICE=='cuda':torch.cuda.empty_cache()
    (FIELD_CACHE/'vectors.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return result


def field_experiment():
    documents=document_fields()
    category=json.loads((V10_CACHE/'category_selection.json').read_text())['winner']['category_boost']
    contextual_path=next(V10_CACHE.glob('query_filter_vectors_*.joblib'))
    context=joblib.load(contextual_path);contextual=dict(zip(context['texts'],context['vectors']))
    del context;gc.collect()
    weights=[.025,.05,.1]
    values={};known=None
    for mode in ['unseen_text','held_context']:
        frame,records,features,reference,known=v9_pool('development',mode)
        reference=compatibility_boost(frame,records,reference,category)
        truth=labels_from_gold(gold,frame)
        values[mode]={('baseline','plain',0.):per_query_recall(top50(records,reference),truth)}
        for name,doc_vectors in documents.items():
            for query_kind in ['plain','with_filters']:
                scores=[]
                for q,(ids,raw) in zip(frame.itertuples(index=False),records):
                    vector=semantic_index.queries[semantic_index.query_to_row[q.query_norm]] if query_kind=='plain' else contextual[
                        query_filter_text(q.query_norm,q.search_infm_params_text)]
                    cosine=(doc_vectors[ids].astype(np.float64)@vector.astype(np.float64)).astype(np.float32)
                    scores.append(semantic_affinity(cosine,raw[:,4]))
                for weight in weights:
                    prediction=top50(records,blend_scores(scores,reference,weight))
                    values[mode][(name,query_kind,weight)]=per_query_recall(prediction,truth)
                del scores;gc.collect()
        del records,features,reference;gc.collect()
    rows=[dict(document=name,query_kind=kind,weight=weight,
        **matched_metrics(values['unseen_text'][(name,kind,weight)],values['held_context'][(name,kind,weight)],known))
        for name,kind,weight in values['unseen_text']]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(FIELD_CACHE/'development.csv',index=False,lineterminator='\n')
    baseline=next(row for row in rows if row['document']=='baseline')
    eligible=table[(table.unseen_macro>=baseline['unseen_macro']-.0005)
        &(table.held_macro>=baseline['held_macro']-.0005)]
    winner=eligible.iloc[0].to_dict()
    report={'winner':winner,'baseline':baseline,'control_used_for_selection':False,
        'grid_weights':weights,'configs':FIELD_CONFIGS,'source_sha256':sha256_file(FIELD_DRIVER),
        'limitations':['Reused development/control; inherited v9 auxiliary-prior limitation.']}
    (FIELD_CACHE/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Document field comparison',table.to_string(index=False),flush=True)
    if winner['document']=='baseline':
        print('No field-channel gain; keep these vectors outside the submission.',flush=True)
        return
    frame,records,features,reference,_=v9_pool('control','unseen_text')
    reference=compatibility_boost(frame,records,reference,category)
    scores=[]
    for q,(ids,raw) in zip(frame.itertuples(index=False),records):
        vector=semantic_index.queries[semantic_index.query_to_row[q.query_norm]] if winner['query_kind']=='plain' else contextual[
            query_filter_text(q.query_norm,q.search_infm_params_text)]
        cosine=(documents[winner['document']][ids].astype(np.float64)@vector.astype(np.float64)).astype(np.float32)
        scores.append(semantic_affinity(cosine,raw[:,4]))
    a=per_query_recall(top50(records,reference),labels_from_gold(gold,frame))
    b=per_query_recall(top50(records,blend_scores(scores,reference,winner['weight'])),labels_from_gold(gold,frame))
    control_report={'baseline_recall50':float(a.mean()),'selected_recall50':float(b.mean()),
        'improved':int((b>a).sum()),'worsened':int((b<a).sum()),'used_for_selection':False}
    (FIELD_CACHE/'control.json').write_text(json.dumps(control_report,indent=2),encoding='utf-8')
    print('Document field control',json.dumps(control_report),flush=True)


if __name__=='__main__':field_experiment()
