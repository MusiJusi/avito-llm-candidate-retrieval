"""Compute the selected upstream ensemble on validation inputs.

Shared by the last cross-encoder combination experiment. Numerical feature
producers are loaded into one existing namespace to keep RAM bounded.
"""
import ast
import gc
import json


def install(namespace):
    root=namespace['ROOT']
    for filename in ['field_ranker_v10.py','bge_ranker_v10.py']:
        path=root/'.development'/filename
        nodes=[node for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    namespace['BGE_CACHE']=root/'artifacts/bge-m3-v10'
    namespace['BGE_DRIVER']=root/'.development/bge_ranker_v10.py'
    field=namespace['install_field_functions']()
    bge_path=namespace['BGE_CACHE']/'selection.json'
    bge=json.loads(bge_path.read_text()) if bge_path.exists() else None
    if bge and bge['winner']['kind']!='baseline':
        namespace['BGE_FP']=bge['fingerprint']
        namespace['BGE_DOCUMENTS'],namespace['BGE_QUERIES']=namespace['prepare_bge']()
    namespace['CONTEXT_VECTORS']=namespace['contextual_vectors']()[0]
    return field,bge


def compute(n,stage,mode,return_warm_features=False):
    root=n['ROOT'];np=n['np'];joblib=n['joblib']
    frame,records,base,reference,known=n['v9_pool'](stage,mode)
    primary=json.loads((n['V10_CACHE']/'selection.json').read_text())
    reference=n['compatibility_boost'](frame,records,reference,primary['category_boost'])
    history=n['evaluation_history'](frame,mode,n['control'] if stage=='development' else n['development'])
    aux_path=n['V10_CACHE']/'warm_aux_selection.json'
    aux=json.loads(aux_path.read_text()) if aux_path.exists() else None
    use_aux=bool(aux and aux['winner']['weight']>0)
    primary_warm=primary['winner']['variant'].startswith('warm_')
    bank_type=n['QualityEvidenceWithTrace'] if primary_warm or use_aux or return_warm_features else n['QualityEvidence']
    n['QUALITY_BANK']=bank_type(history);del history;gc.collect()
    full=n['extra_matrices'](frame,records,base)
    x=full if primary_warm else [v[:,:82] for v in full]
    for q,values in zip(frame.itertuples(index=False),x):
        if float(q.search_category)==0:values[:,70]=1.
    name=primary['winner']['variant']
    if name in n['MODEL_RECIPES']:
        recipe=n['MODEL_RECIPES'][name]
        model=joblib.load(n['V10_CACHE']/f'evaluation_{name}_{n["V10_FP"]}.joblib')
        reference=n['blend_scores'](n['predict_scores'](model,[v[:,recipe['columns']] for v in x],recipe['trees']),
            reference,primary['winner']['weight'])
        del model
    field=json.loads((root/'artifacts/field-ranker-v10/selection.json').read_text())
    if field['winner']['weight']>0:
        x=[np.column_stack([values,n['field_features'](q,ids,values)]).astype(np.float32)
            for q,(ids,_),values in zip(frame.itertuples(index=False),records,x)]
        model=joblib.load(n['FIELD_RANK_CACHE']/f'evaluation_ranker_{n["FIELD_RANK_FP"]}.joblib')
        reference=n['blend_scores'](n['predict_scores'](model,x,500),reference,field['winner']['weight']);del model
    bge_path=n['BGE_CACHE']/'selection.json'
    bge=json.loads(bge_path.read_text()) if bge_path.exists() else None
    if bge and bge['winner']['kind']!='baseline':
        extra=[n['bge_features'](q,ids,values) for q,(ids,_),values in zip(frame.itertuples(index=False),records,x)]
        if bge['winner']['kind']=='ranker':
            augmented=[np.column_stack([values,e]).astype(np.float32) for values,e in zip(x,extra)]
            model=joblib.load(n['bge_model_path']('evaluation',bge['winner']['variant']))
            prediction=n['predict_scores'](model,augmented,int(bge['trees']));del augmented,model
        else:prediction=[n['semantic_affinity'](e[:,0],raw[:,4]) for e,(_,raw) in zip(extra,records)]
        reference=n['blend_scores'](prediction,reference,bge['winner']['weight']);del prediction,extra
    if use_aux:
        name=aux['winner']['variant'];recipe=n['MODEL_RECIPES'][name]
        model=joblib.load(n['V10_CACHE']/f'evaluation_{name}_{n["V10_FP"]}.joblib')
        reference=n['blend_scores'](n['predict_scores'](model,full,recipe['trees']),reference,aux['winner']['weight'])
        del model
    del base,x;gc.collect()
    if return_warm_features:return frame,records,reference,known,full
    return frame,records,reference,known
