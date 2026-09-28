"""Fallback resumable download of the same pinned official public checkpoint.

Bounded ranges avoid long-lived CDN transfers on an unreliable connection.
The official LFS SHA-256 must match before publishing model weights.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import concurrent.futures
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.semantic_deps'))
import requests

REVISION='5617a9f61b028005a4858fdac845db406aefb181'
EXPECTED='b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38'


def download():
    directory=ROOT/'models/bge-m3'
    partial=ROOT/'.downloads/bge-m3-ranges.bin';partial.parent.mkdir(exist_ok=True)
    url=f'https://huggingface.co/BAAI/bge-m3/resolve/{REVISION}/pytorch_model.bin?download=true'
    chunk=8*1024*1024;offset=partial.stat().st_size if partial.exists() else 0
    total=None;local=threading.local()
    def fetch(start):
        if not hasattr(local,'session'):local.session=requests.Session()
        end=start+chunk-1
        for attempt in range(5):
            try:
                with local.session.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,45),stream=True) as response:
                    assert response.status_code==206,f'Expected range response; HTTP {response.status_code}'
                    range_header=response.headers['Content-Range']
                    observed=int(range_header.split(' ')[1].split('-')[0]);size=int(range_header.split('/')[1])
                    assert observed==start,'CDN returned a different range'
                    block=response.content
                    assert len(block)==min(chunk,size-start)
                return block,size
            except Exception as error:
                print('Range retry',start,attempt+1,type(error).__name__,flush=True)
                if attempt==4:raise
                time.sleep(2)
    block,total=fetch(offset)
    with partial.open('ab') as output:output.write(block)
    offset+=len(block)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while offset<total:
            starts=range(offset,min(offset+4*chunk,total),chunk)
            futures=[pool.submit(fetch,start) for start in starts]
            # Publish in byte order; an interrupted batch resumes at the last
            # fully written range. At most four small blocks are held in RAM.
            for future in futures:
                block,size=future.result();assert size==total
                with partial.open('ab') as output:output.write(block)
                offset+=len(block)
            print('BGE weight bytes',offset,total,flush=True)
    hasher=hashlib.sha256()
    with partial.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):hasher.update(block)
    assert hasher.hexdigest()==EXPECTED,'Official LFS hash mismatch'
    partial.replace(directory/'pytorch_model.bin')
    names=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
        'sentencepiece.bpe.model','README.md','1_Pooling/config.json','modules.json','pytorch_model.bin']
    hashes={}
    for name in names:
        digest=hashlib.sha256()
        with (directory/name).open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
        hashes[name]=digest.hexdigest()
    manifest={'model':'BAAI/bge-m3','revision':REVISION,'license':'MIT',
        'source':'https://huggingface.co/BAAI/bge-m3','files_sha256':hashes,
        'task_data_sent':False,'external_inference_api_used':False,'download':'Official CDN byte ranges, LFS SHA checked'}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Pinned BGE-M3 downloaded and hash verified',flush=True)


if __name__=='__main__':download()
