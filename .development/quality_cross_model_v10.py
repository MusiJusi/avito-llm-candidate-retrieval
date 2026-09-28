"""Local cross-encoder inference for the selected v10 combination.

The notebook actually evaluates the model on texts; no benchmark answer/logit
cache is used. CUDA BF16 matches the validation pilot; CPU float32 is a slower
fallback. The delivery proof records which device actually reproduced the CSV.
"""
import gc
import numpy as np
import pandas as pd
from quality_signals_v10 import query_filter_text


def adjust_cross_scores(root,manifest,queries,items,records,scores,v9_scores,
                        sha256_file,blend_scores,stable_topk):
    if not manifest.get('cross_weight'):return scores
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    assert sha256_file(root/manifest['cross_weights'])==manifest['cross_weights_sha256']
    for filename,expected in manifest['cross_files_sha256'].items():assert sha256_file(root/filename)==expected
    tokenizer=AutoTokenizer.from_pretrained(root/manifest['cross_tokenizer'],local_files_only=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModelForSequenceClassification.from_pretrained(root/manifest['cross_model'],
        local_files_only=True,attn_implementation='eager').to(device).float().eval()
    documents=(items.item_title_raw+'. '+items.item_infm_params_text.str.slice(0,500)+
        '. '+items.item_description_raw.str.slice(0,1800)).to_numpy()
    categories=pd.to_numeric(items.item_category_id,errors='coerce').to_numpy()
    outputs=[]
    with torch.inference_mode():
        for row,(q,(ids,_),base) in enumerate(zip(queries.itertuples(index=False),records,v9_scores)):
            category=float(q.search_category)
            reference=base+float(manifest['category_boost'])*((category>0)&(categories[ids]==category))
            selected=stable_topk(reference,200)
            query=query_filter_text(q.query_norm,q.search_infm_params_text)
            result=np.full(len(ids),-1e6,np.float32)
            for start in range(0,len(selected),64):
                positions=selected[start:start+64]
                batch=tokenizer([query]*len(positions),documents[ids[positions]].tolist(),
                    truncation=True,padding=True,max_length=192,return_tensors='pt')
                batch={k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
                    result[positions]=model(**batch).logits.flatten().float().cpu().numpy()
            outputs.append(result)
            if row%100==0:print('Local cross-encoder',device,row,len(queries),flush=True)
    del model,tokenizer,documents;gc.collect()
    return blend_scores(outputs,scores,float(manifest['cross_weight']))
