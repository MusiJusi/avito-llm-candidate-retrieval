"""Measure full-corpus retrieval with the learned E5, before retraining a ranker.

This diagnostic never injects positive items into candidates. The document
vectors stay frozen. The development query encoder excludes all evaluation
texts, and geographic histories use the exact prior evaluation exclusions.
Pool recall is reported separately from final Recall@50: a larger pool is not
itself an improved submission.
"""
from pathlib import Path
import ast
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
POOL_CACHE = ROOT/'artifacts/learned-pool-v8'
POOL_CACHE.mkdir(exist_ok=True)

def measure_pool():
    vectors = evaluation_query_features(development,'development')
    lookup = {text:i for i,text in enumerate(sorted(set(development.query_norm)))}
    truths = labels_from_gold(gold,development)
    documents = torch.tensor(semantic_index.items,device=DEVICE)
    rows = []
    details = []
    for mode in ['unseen_text','held_context']:
        records,features,known,audit = evaluation_features(development,mode,'development',control)
        del features
        gc.collect()
        fit = evaluation_history(development,mode,control)
        history = V5History(fit)
        del fit
        geo_cache = {}
        output = {k:[] for k in [0,500,1000]}
        counts = {k:[] for k in output}
        rescued = {k:0 for k in output}
        outside = 0
        started = time.perf_counter()
        for start in range(0,len(development),64):
            frame = development.iloc[start:start+64]
            query_vectors = torch.tensor(np.stack([vectors[lookup[t]] for t in frame.query_norm]),device=DEVICE)
            with torch.inference_mode():
                block = (query_vectors@documents.T).cpu().numpy()
            del query_vectors
            for offset,query in enumerate(frame.itertuples(index=False)):
                position = start+offset
                original = records[position][0]
                truth = truths[position]
                missing = set(truth)-set(original)
                outside += len(missing)
                loc = int(query.search_location_id)
                if loc not in geo_cache:
                    if len(geo_cache)>=64:geo_cache.clear()
                    geo_cache[loc] = history.geography(loc)
                cosine = block[offset]
                geographic = semantic_affinity(cosine,geo_cache[loc])
                pure = stable_topk(cosine,1000)
                geo = stable_topk(geographic,1000)
                for k in output:
                    combined = original if k==0 else np.union1d(original,np.union1d(pure[:k],geo[:k]))
                    caught = set(combined)&truth
                    output[k].append(len(caught)/len(truth))
                    counts[k].append(len(combined))
                    rescued[k] += len(caught&missing)
                if missing:
                    details.append({'mode':mode,'context_key':query.context_key,'query':query.query_norm,
                        'missing_pairs':len(missing),'rescued500':len(missing&set(np.union1d(pure[:500],geo[:500]))),
                        'rescued1000':len(missing&set(np.union1d(pure,geo)))})
            if start%640==0:print('Learned full-corpus pool',mode,start,'seconds',round(time.perf_counter()-started,1),flush=True)
        for k in output:
            rows.append({'mode':mode,'added_per_channel':k,'pool_recall':float(np.mean(output[k])),
                'mean_pool_size':float(np.mean(counts[k])),'outside_positive_pairs':outside,
                'rescued_positive_pairs':rescued[k],'own_context_overlap':audit['own_context_overlap']})
        del records,history,geo_cache,block
        gc.collect()
    pd.DataFrame(rows).to_csv(POOL_CACHE/'development.csv',index=False)
    pd.DataFrame(details).to_csv(POOL_CACHE/'rescued_queries.csv',index=False)
    report = {'results':rows,'main_answer_unchanged':sha256_file(ROOT/'answer.csv')==BASE_ANSWER_HASH,
        'control_evaluated':False,'limitation':'Candidate pool only; a trained selector must validate Recall@50.'}
    assert report['main_answer_unchanged']
    (POOL_CACHE/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Learned pool diagnostic',json.dumps(report),flush=True)

if __name__=='__main__':
    measure_pool()
