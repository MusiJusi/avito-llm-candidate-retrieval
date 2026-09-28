"""History-availability calibration for item-popularity features.

The strict cached query encoder remains fixed. New feature histories are
supersets of its permitted history, still excluding all own text/context labels.
One cold-history group per context plus warm cohorts avoids treating deliberate
positive-item erasure as a permanent negative popularity signal. Category
dropout represents unspecified searches without changing positive labels.
"""
from pathlib import Path
import ast
import gc
import hashlib
import json
import os
import time

WARM_DRIVER=Path(__file__).resolve()
source=WARM_DRIVER.with_name('train_quality_v10.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
__file__=str(source)
exec(compile(tree,str(source),'exec'),globals())
__file__=str(WARM_DRIVER)
from quality_trace_v10 import QueryItemEvidence,QualityEvidenceWithTrace,TRACE_NAMES
BASE_V10_FP=V10_FP
BASE_V10_CACHE=V10_CACHE
WARM_CACHE=ROOT/'artifacts/warm-history-v10';WARM_CACHE.mkdir(exist_ok=True)
WARM_CONFIG={'evaluation_contexts':6000,'final_contexts':8000,'warm_weight':.4,
    'fractions':[0.,.5],'cold_groups_per_context':1,'trees':500,
    'category_dropout_fraction':float(queries.search_category.eq(0).mean()),'seed':SEED+2201}
WARM_FP=hashlib.sha256(json.dumps({'base':BASE_V10_FP,'config':WARM_CONFIG,
    'source':sha256_file(WARM_DRIVER),'trace':sha256_file(WARM_DRIVER.with_name('quality_trace_v10.py'))},sort_keys=True).encode()).hexdigest()[:16]
WARM_RECIPE={'columns':list(range(86)),'trees':500,'leaves':31,'min_child':80}
MODEL_RECIPES['warm_mixed']=WARM_RECIPE


def profile_fp(fraction):
    return hashlib.sha256(f'{WARM_FP}:{fraction:g}'.encode()).hexdigest()[:16]


def generate_warm(stage,fraction):
    global V10_CACHE,V10_FP
    fp=profile_fp(fraction)
    if (WARM_CACHE/f'{stage}_training_{fp}.json').exists():return
    # The independent miners have exactly the same labels, feature columns and
    # text assignments. Reuse immutable models, not warm feature matrices.
    originals=independent_miners(stage)
    for fold in range(3):
        old=BASE_V10_CACHE/f'{stage}_textheld_miner_{fold}_{BASE_V10_FP}.joblib'
        new=WARM_CACHE/f'{stage}_textheld_miner_{fold}_{fp}.joblib'
        if not new.exists():os.link(old,new)
    del originals;gc.collect()
    base_source=WARM_DRIVER.with_name('train_quality_v10.py')
    tree=ast.parse(base_source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_quality_training')
    text=ast.get_source_segment(base_source.read_text(encoding='utf-8'),node)
    text=text.replace('def prepare_quality_training(', 'def prepare_warm_data(')
    marker="            assert bundle['history_sha256']==history_digest(history)"
    replacement=marker+'''
            strict_history=history
            _,strict_cold,_=history_for_queries(source_history,subset,[truth[i] for i in positions],
                ITEM_IDS,mode,.9,SEED+804+fold)
            all_ids={ITEM_IDS[i] for t in [truth[p] for p in positions] for i in t}
            blocked=set() if WARM_FRACTION==0 else set(np.random.default_rng(SEED+2210+fold).choice(
                sorted(strict_cold),int(WARM_FRACTION*len(all_ids)),replace=False))
            history,_,_=history_for_queries(source_history,subset,[truth[i] for i in positions],
                ITEM_IDS,mode,0.,SEED+804+fold,excluded_items=blocked)
            assert strict_history.index.isin(history.index).all()
            del strict_history
            chosen=selected_training(frame,WARM_CONFIG[stage+'_contexts'])
            picked={key for key in chosen if (int(hashlib.sha256(('warm:'+key).encode()).hexdigest()[:8],16)%2==0)==(WARM_FRACTION==0)}
            subset=subset[subset.context_key.isin(picked)]
'''
    assert text.count(marker)==1
    text=text.replace(marker,replacement)
    text=text.replace("audit.update(mode=mode,fold=fold)",
        "audit.update(mode=mode,fold=fold,feature_cold_fraction=WARM_FRACTION,encoder_history_is_subset=True)")
    exec(compile(text,str(WARM_DRIVER),'exec'),globals())
    old_cache,old_fp=V10_CACHE,V10_FP
    V10_CACHE,V10_FP=WARM_CACHE,fp
    globals()['WARM_FRACTION']=fraction
    try:
        data=prepare_warm_data(stage)
        del data;gc.collect()
        path=WARM_CACHE/f'{stage}_training_{fp}.json'
        report=json.loads(path.read_text());report.update(warm_fraction=fraction,
            source_sha256=sha256_file(WARM_DRIVER),strict_encoder_history_is_subset=True)
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:V10_CACHE,V10_FP=old_cache,old_fp


def ordered_paths(directory,stage,fp):
    for mode in ['unseen_text','held_context']:
        for fold in range(3):
            prefix=f'{stage}_{mode}_{fold}_'
            paths=list(directory.glob(prefix+f'*_{fp}.joblib'))
            paths.sort(key=lambda p:int(p.stem[len(prefix):].split('_')[0]))
            for path in paths:yield mode,path


def mixed_groups(stage,with_features=True):
    """The same iterator defines the mixed ranker and field-feature ablation."""
    frame,source_history,_,_=frame_for_stage(stage)
    truth=labels_from_gold(gold,frame);frame_by_key=frame.set_index('context_key')
    for cohort,fraction,directory,fp in [('cold',.9,BASE_V10_CACHE,BASE_V10_FP),
        ('warm',0.,WARM_CACHE,profile_fp(0.)),('warm',.5,WARM_CACHE,profile_fp(.5))]:
        current=None;trace=None
        for mode,path in ordered_paths(directory,stage,fp):
            fold=int(path.stem.split('_')[3])
            if with_features and current!=(mode,fold):
                assignments=oof_assignments(frame,mode,3,SEED+803)
                positions=np.flatnonzero(assignments==fold);held=frame.iloc[positions]
                held_truth=[truth[i] for i in positions]
                history,strict_cold,_=history_for_queries(source_history,held,held_truth,ITEM_IDS,mode,.9,SEED+804+fold)
                if fraction!=.9:
                    all_ids={ITEM_IDS[i] for t in held_truth for i in t}
                    blocked=set() if fraction==0 else set(np.random.default_rng(SEED+2210+fold).choice(
                        sorted(strict_cold),int(fraction*len(all_ids)),replace=False))
                    history,_,_=history_for_queries(source_history,held,held_truth,ITEM_IDS,mode,0.,SEED+804+fold,excluded_items=blocked)
                trace=QueryItemEvidence(history);del history;gc.collect()
                current=mode,fold
            for x,y,known,key,text,ids in joblib.load(path):
                chosen_mode='unseen_text' if int(hashlib.sha256(('mode:'+key).encode()).hexdigest()[:8],16)%2==0 else 'held_context'
                if cohort=='cold' and mode!=chosen_mode:continue
                if with_features:
                    x=x.copy()
                    dropout=int(hashlib.sha256(('category:'+key).encode()).hexdigest()[:8],16)/2**32
                    if dropout<WARM_CONFIG['category_dropout_fraction']:x[:,70]=1.
                    q=frame_by_key.loc[key]
                    if float(q.search_category)==0:x[:,70]=1.
                    x=np.column_stack([x,trace.features(q,ITEM_IDS[ids])]).astype(np.float32)
                yield x,y,known,key,text,ids,cohort=='warm'


def mixed_training(stage):
    for fraction in [0.,.5]:generate_warm(stage,fraction)
    path=WARM_CACHE/f'{stage}_mixed_{WARM_FP}.json'
    if not path.exists():
        sizes=[];known=[];warm=[];keys=[]
        for x,y,k,key,text,ids,is_warm in mixed_groups(stage,with_features=False):
            sizes.append(len(y));known.append(k);warm.append(is_warm);keys.append(key)
        assert sizes
        rows=sum(sizes)
        x_path=WARM_CACHE/f'{stage}_mixed_X_{WARM_FP}.npy';y_path=WARM_CACHE/f'{stage}_mixed_y_{WARM_FP}.npy'
        x=np.lib.format.open_memmap(x_path,mode='w+',dtype=np.float32,shape=(rows,86))
        y=np.lib.format.open_memmap(y_path,mode='w+',dtype=np.uint8,shape=(rows,))
        offset=0
        for features,target,*_ in mixed_groups(stage):
            stop=offset+len(target);x[offset:stop]=features;y[offset:stop]=target;offset=stop
        assert offset==rows
        x.flush();y.flush();del x,y;gc.collect()
        known=np.asarray(known,bool);warm=np.asarray(warm,bool);weights=np.empty(len(sizes),np.float32)
        for cohort,share in [(False,1-WARM_CONFIG['warm_weight']),(True,WARM_CONFIG['warm_weight'])]:
            mask=warm==cohort;observed=known[mask].mean()
            group=np.where(known[mask],known_target/max(observed,1e-6),(1-known_target)/max(1-observed,1e-6))
            weights[mask]=group*share*len(sizes)/mask.sum()
        report={'fingerprint':WARM_FP,'stage':stage,'rows':rows,'groups':len(sizes),
            'warm_groups':int(warm.sum()),'covered_contexts':len(set(keys)),
            'sizes':sizes,'group_weight':weights.tolist(),'x_file':x_path.name,'y_file':y_path.name,
            'config':WARM_CONFIG,'source_sha256':sha256_file(WARM_DRIVER)}
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Mixed warm history dataset',stage,rows,len(sizes),flush=True)
    report=json.loads(path.read_text())
    return {'X':np.load(WARM_CACHE/report['x_file'],mmap_mode='r'),'y':np.load(WARM_CACHE/report['y_file'],mmap_mode='r'),
        'sizes':np.asarray(report['sizes'],np.int32),'group_weight':np.asarray(report['group_weight'],np.float32),'report':report}


def compare_warm():
    global RESOURCES,CONTEXT_VECTORS
    install_retriever();RESOURCES=prepare_resources();CONTEXT_VECTORS,_=contextual_vectors()
    data=mixed_training('evaluation')
    model=fit_quality(data,'evaluation',['warm_mixed'])['warm_mixed'];del data;gc.collect()
    globals()['QualityEvidence']=QualityEvidenceWithTrace
    category=json.loads((BASE_V10_CACHE/'category_selection.json').read_text())['winner']['category_boost']
    a,_=evaluate_quality('development','unseen_text',{'warm_mixed':model},category)
    b,known=evaluate_quality('development','held_context',{'warm_mixed':model},category)
    rows=[dict(variant=name,weight=weight,**matched_metrics(a[(name,weight)],b[(name,weight)],known)) for name,weight in a]
    table=pd.DataFrame(rows).sort_values('matched_recall50',ascending=False)
    table.to_csv(WARM_CACHE/'development.csv',index=False,lineterminator='\n')
    selection=json.loads((BASE_V10_CACHE/'selection.json').read_text());previous=dict(selection['winner'])
    best=table[(table.unseen_macro>=previous['unseen_macro']-.0005)
        &(table.held_macro>=previous['held_macro']-.0005)].iloc[0].to_dict() if len(table[
        (table.unseen_macro>=previous['unseen_macro']-.0005)&(table.held_macro>=previous['held_macro']-.0005)]) else previous
    report={'winner':best,'previous_winner':previous,'fingerprint':WARM_FP,
        'cold_development_gain':float(best['matched_recall50']-previous['matched_recall50']),
        'control_used_for_selection':False,'config':WARM_CONFIG}
    (WARM_CACHE/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if best['matched_recall50']>previous['matched_recall50']:
        selection['winner_before_warm_training']=previous;selection['winner']=best
        selection['warm_training_fingerprint']=WARM_FP
        (BASE_V10_CACHE/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    c,_=evaluate_quality('control','unseen_text',{'warm_mixed':model},category)
    selected=best if best['variant']=='warm_mixed' else table[table.variant=='warm_mixed'].iloc[0].to_dict()
    (WARM_CACHE/'control.json').write_text(json.dumps({'selected_recall50':float(c[(selected['variant'],selected['weight'])].mean()),
        'used_for_selection':False,'primary_selection_updated':selection['winner']['variant']=='warm_mixed'},indent=2),encoding='utf-8')
    print('Warm history comparison',table.to_string(index=False),flush=True)


if __name__=='__main__':
    import sys
    if '--prepare-only' in sys.argv:
        install_retriever();RESOURCES=prepare_resources();CONTEXT_VECTORS,_=contextual_vectors()
        data=mixed_training('evaluation')
        print('Warm evaluation features ready',WARM_FP,data['X'].shape,flush=True)
    else:compare_warm()
