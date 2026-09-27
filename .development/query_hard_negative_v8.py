"""A bounded continuation of E5 with explicit, semi-hard document negatives.

The checkpoint and all mined documents use only the permitted training history.
All known positives are excluded from negative sampling. Other documents are
unjudged, so the auxiliary loss receives a conservative weight rather than
assuming that every nearest neighbour is an unequivocally negative service.
This is a pilot: downstream scores use the fixed v7 ranker, not a refitted ranker.
"""
from pathlib import Path
import ast
import hashlib
import json
import gc
import time

DRIVER = Path(__file__).resolve()
source = DRIVER.with_name('query_encoder_rank.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
tree.body = [n for n in tree.body if not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__ = str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__ = str(DRIVER)
HN_CACHE = ROOT/'artifacts/query-hard-negative-v8'
HN_CACHE.mkdir(exist_ok=True)
HN_CONFIG = dict(epochs=1, learning_rate=1e-5, negatives=4, retrieve=64,
    skip_nearest=8, hard_weight=.25, seed=SEED+1002)
HN_FP = hashlib.sha256(json.dumps({'source':hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
    'query_source':QE_FP,'config':HN_CONFIG,'history':history_digest(training_history)},
    sort_keys=True).encode()).hexdigest()[:16]

def fit_hard_encoder():
    checkpoint = HN_CACHE/f'encoder_{HN_FP}'
    vector_path = HN_CACHE/f'development_{HN_FP}.npy'
    if (checkpoint/'model.safetensors').exists() and vector_path.exists():
        return np.load(vector_path,allow_pickle=False)
    assert DEVICE=='cuda','This bounded neural pilot requires the local GPU'
    ids, documents = training_documents()
    positives = training_history.groupby('query_norm',sort=True).item_id.agg(lambda x:sorted(set(x))).to_dict()
    texts = sorted(positives)
    assert not set(texts)&set(all_evaluation.query_norm)
    item_rows = {item:i for i,item in enumerate(ids)}
    positive_sets = {query:set(values) for query,values in positives.items()}
    model = AutoModel.from_pretrained(QE_CACHE/f'epoch_3_{QE_FP}',local_files_only=True,
        attn_implementation='eager').to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    random = np.random.default_rng(HN_CONFIG['seed'])
    torch.manual_seed(HN_CONFIG['seed'])
    fixed_documents = torch.tensor(documents,device=DEVICE)
    learned_queries = encode_learned(model,tokenizer,texts)
    negatives = np.empty((len(texts),HN_CONFIG['negatives']),np.int64)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(texts),128):
            vectors = torch.tensor(learned_queries[start:start+128],device=DEVICE)
            cosine = vectors@fixed_documents.T
            neighbours = torch.topk(cosine,HN_CONFIG['retrieve'],dim=1).indices.cpu().numpy()
            for offset,rows in enumerate(neighbours):
                query = texts[start+offset]
                eligible = [int(row) for row in rows[HN_CONFIG['skip_nearest']:] if ids[row] not in positive_sets[query]]
                if len(eligible)<HN_CONFIG['negatives']:
                    eligible = [int(row) for row in rows if ids[row] not in positive_sets[query]]
                assert len(eligible)>=HN_CONFIG['negatives']
                negatives[start+offset] = random.choice(eligible,HN_CONFIG['negatives'],replace=False)
                assert not {ids[row] for row in negatives[start+offset]}&positive_sets[query]
            del vectors,cosine
    del learned_queries
    print('Hard negatives mined',negatives.shape,'seconds',round(time.perf_counter()-started,1),flush=True)
    space = prepare_micro_space()
    lookup = {}
    for row,query in enumerate(space['inputs']['query']):
        lookup.setdefault(query,row)
    teacher = np.stack([space['dense'][lookup[query],:384] for query in texts])
    del space,lookup,documents
    gc.collect()
    fixed_teacher = torch.tensor(teacher,device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=HN_CONFIG['learning_rate'],weight_decay=.01)
    model.train()
    order = random.permutation(len(texts))
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0,len(order),128):
        rows = order[start:start+128]
        queries_batch = [texts[i] for i in rows]
        positive_ids = [positives[q][int(random.integers(len(positives[q])))] for q in queries_batch]
        positive_rows = torch.tensor([item_rows[item] for item in positive_ids],device=DEVICE)
        target = torch.tensor(in_batch_targets(queries_batch,positive_ids,positive_sets),device=DEVICE)
        batch = tokenizer(['query: '+q for q in queries_batch],padding=True,truncation=True,
            max_length=64,return_tensors='pt')
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            vector = torch.nn.functional.normalize(((hidden*mask).sum(1)/mask.sum(1).clamp(min=1)).float(),p=2,dim=1)
        in_batch = vector@fixed_documents[positive_rows].T/.05
        ordinary = -(target*torch.nn.functional.log_softmax(in_batch,dim=1)).sum(1).mean()
        negative_rows = torch.tensor(negatives[rows],device=DEVICE)
        positive_score = (vector*fixed_documents[positive_rows]).sum(1,keepdim=True)
        negative_score = torch.einsum('bd,bnd->bn',vector,fixed_documents[negative_rows])
        explicit = torch.cat([positive_score,negative_score],dim=1)/.05
        hard = torch.nn.functional.cross_entropy(explicit,torch.zeros(len(rows),dtype=torch.long,device=DEVICE))
        retention = (1-(vector*fixed_teacher[rows]).sum(1)).mean()
        loss = ordinary+HN_CONFIG['hard_weight']*hard+.2*retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if start%(128*100)==0:
            print('Hard-negative training',start,'/',len(texts),'loss',round(float(np.mean(losses[-100:])),4),flush=True)
    checkpoint.mkdir(exist_ok=True)
    model.save_pretrained(checkpoint,safe_serialization=True)
    vectors = encode_learned(model,tokenizer,sorted(set(development.query_norm)))
    save_array(vectors,vector_path)
    report = {'fingerprint':HN_FP,'config':HN_CONFIG,'training_pairs_available':len(training_history),
        'sampled_pairs':len(texts),'development_text_overlap':0,'known_positives_in_negatives':0,
        'initial_checkpoint':str((QE_CACHE/f'epoch_3_{QE_FP}').relative_to(ROOT)),
        'mean_loss':float(np.mean(losses)),'seconds':round(time.perf_counter()-started,2),
        'gpu_peak_allocated_MiB':round(torch.cuda.max_memory_allocated()/2**20,1),
        'limitation':'Mined non-selected items remain unjudged; conservative auxiliary loss.'}
    (HN_CACHE/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    del model,optimizer,fixed_documents,fixed_teacher,teacher
    gc.collect()
    torch.cuda.empty_cache()
    return vectors

def evaluate_hard_encoder(vectors):
    lookup = {text:i for i,text in enumerate(sorted(set(development.query_norm)))}
    original_vectors = evaluation_query_features(development,'development')
    ranker = joblib.load(QR_CACHE/f'evaluation_ranker_{QR_FP}.joblib')
    recalls = {}
    for mode in ['unseen_text','held_context']:
        records,features,known,audit = evaluation_features(development,mode,'development',control)
        baseline = baseline_v5(records,features)
        prediction_scores = []
        for vector_set in [original_vectors,vectors]:
            matrices = [np.column_stack([x,query_features(vector_set[lookup[text]],ids,raw)])
                for text,(ids,raw),x in zip(development.query_norm,records,features)]
            prediction_scores.append(blend_scores(predict_scores(ranker,matrices,400),baseline,.75))
            del matrices
        truth = labels_from_gold(gold,development)
        for weight in [0.,.25,.5,1.]:
            prediction = top50(records, prediction_scores[0] if weight==0 else
                blend_scores(prediction_scores[1],prediction_scores[0],weight))
            recalls[(mode,weight)] = per_query_recall(prediction,truth)
        if mode=='held_context':held_known=known
        del records,features,baseline,prediction_scores
        gc.collect()
    query_known = queries.query_norm.isin(history_all.query_norm)
    rows = []
    for weight in [0.,.25,.5,1.]:
        a = recalls[('unseen_text',weight)];b=recalls[('held_context',weight)]
        score = (1-known_target)*matched_slice_recall(development,a,queries[~query_known])+known_target*matched_slice_recall(development,b,queries[query_known],held_known)
        rows.append({'hard_encoder_weight':weight,'matched_recall50':score,
            'unseen_recall50':float(a.mean()),'held_recall50':float(b.mean())})
    table = pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(HN_CACHE/'development.csv',index=False)
    assert abs(table[table.hard_encoder_weight==0].iloc[0].matched_recall50-.9501708535101647)<1e-10
    report = {'fingerprint':HN_FP,'winner':table.iloc[0].to_dict(),'control_evaluated':False,
        'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'limitation':'Fixed v7 ranker; positive result requires isolated OOF features and refit.'}
    assert report['main_answer_unchanged']
    (HN_CACHE/'pilot_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Hard-negative pilot',json.dumps(report),flush=True)

if __name__=='__main__':
    evaluate_hard_encoder(fit_hard_encoder())
