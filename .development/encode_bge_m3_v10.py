"""Local, pinned BGE-M3 dense embeddings; no task data leaves the laptop.

CLS pooling and normalization follow the official model card. E5 remains
available: this is an independent signal, not a destructive encoder migration.
An interrupted encoding resumes at completed length-sorted batches.
"""
from pathlib import Path
import ast
import ctypes
import gc
import hashlib
import json
import os
import sys
import time

ENCODE_DRIVER=Path(__file__).resolve()
ROOT=ENCODE_DRIVER.parents[1]


def wait_pid(pid):
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(ctypes.c_void_p(handle),10000)==258:pass
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))


def main():
    (ROOT/'artifacts/quality-v10/bge_process.json').write_text(json.dumps({
        'encoder_pid':os.getpid(),'workspace':str(ROOT)}),encoding='utf-8')
    if len(sys.argv)>1:wait_pid(int(sys.argv[1]))
    checkpoint=ROOT/'models/bge-m3/manifest.json'
    until=time.monotonic()+5400
    while not checkpoint.exists() and time.monotonic()<until:time.sleep(10)
    assert checkpoint.exists(),'Pinned model download did not complete in time'
    source=ENCODE_DRIVER.with_name('train_quality_v10.py')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not(isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
    # Frozen joblib indexes contain __main__.BM25Index. Their class definitions
    # must live in this actual module, not in an isolated execution dictionary.
    namespace=globals();namespace['__file__']=str(source)
    exec(compile(tree,str(source),'exec'),namespace)
    namespace['__file__']=str(ENCODE_DRIVER)
    np=namespace['np'];pd=namespace['pd'];torch=namespace['torch'];joblib=namespace['joblib']
    from transformers import AutoModel,AutoTokenizer
    model_manifest=json.loads(checkpoint.read_text())
    cache=ROOT/'artifacts/bge-m3-v10';cache.mkdir(exist_ok=True)
    config={'revision':model_manifest['revision'],'doc_max_length':256,'query_max_length':128,
        'batch_size':32,'pooling':'normalized CLS','params_chars':1000,'description_chars':2400,
        'source_sha256':namespace['sha256_file'](ENCODE_DRIVER),
        'input_sha256':namespace['input_hashes'],'dtype':'bfloat16 encoder, normalized float32 vectors'}
    fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/bge-m3',local_files_only=True)
    assert namespace['DEVICE']=='cuda'
    model=AutoModel.from_pretrained(ROOT/'models/bge-m3',local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    def encode(texts,name,length):
        path=cache/f'{name}_{fp}.npy'
        if path.exists():return path
        partial=path.with_suffix('.npy.partial');progress=path.with_suffix('.progress.json')
        order=np.argsort(np.asarray([len(t) for t in texts]),kind='stable')
        matrix=np.lib.format.open_memmap(partial,mode='r+' if progress.exists() else 'w+',
            dtype=np.float32,shape=(len(texts),1024))
        completed=json.loads(progress.read_text())['completed'] if progress.exists() else 0
        started=time.perf_counter()
        with torch.inference_mode():
            for start in range(completed,len(texts),config['batch_size']):
                ids=order[start:start+config['batch_size']]
                batch=tokenizer([texts[i] for i in ids],padding=True,truncation=True,
                    max_length=length,return_tensors='pt')
                batch={k:v.to('cuda') for k,v in batch.items()}
                vector=torch.nn.functional.normalize(model(**batch).last_hidden_state[:,0].float(),dim=1)
                matrix[ids]=vector.cpu().numpy()
                stop=min(start+config['batch_size'],len(texts))
                if stop%4096==0 or stop==len(texts):
                    matrix.flush()
                    progress.write_text(json.dumps({'completed':stop,'total':len(texts),'fingerprint':fp}),encoding='utf-8')
                    print('BGE embeddings',name,stop,len(texts),'seconds',round(time.perf_counter()-started),flush=True)
        matrix.flush();del matrix;gc.collect();partial.replace(path)
        return path
    items=namespace['items']
    documents=(items.item_title_raw+'. '+items.item_infm_params_text.str.slice(0,1000)+
        '. '+items.item_description_raw.str.slice(0,2400)).tolist()
    doc_path=encode(documents,'documents',256)
    del documents;gc.collect()
    frame=pd.concat([namespace['history_all'][['query_norm','search_infm_params_text']],
        namespace['queries'][['query_norm','search_infm_params_text']]],ignore_index=True).drop_duplicates()
    make_text=namespace['query_filter_text']
    texts=sorted(set(frame.query_norm)|{make_text(q,f) for q,f in frame.itertuples(index=False,name=None)})
    query_path=encode(texts,'queries',128)
    texts_path=cache/f'query_texts_{fp}.joblib';joblib.dump(texts,texts_path)
    report={'fingerprint':fp,'config':config,'model':model_manifest,
        'documents':{'path':doc_path.relative_to(ROOT).as_posix(),'sha256':namespace['sha256_file'](doc_path)},
        'queries':{'path':query_path.relative_to(ROOT).as_posix(),'sha256':namespace['sha256_file'](query_path)},
        'query_texts':{'path':texts_path.relative_to(ROOT).as_posix(),'sha256':namespace['sha256_file'](texts_path)},
        'task_data_sent':False,'external_inference_api_used':False}
    (cache/'vectors.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('BGE-M3 encoding complete',fp,flush=True)


if __name__=='__main__':main()
