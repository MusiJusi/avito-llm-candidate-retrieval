"""Download a pinned public encoder checkpoint; never upload task data.

Only model files are fetched. Inference is local, and anonymous download avoids
using any account token. Encoding/quality testing runs separately after GPU is
free. Official architecture/pooling configuration is retained with provenance.
"""
from pathlib import Path
import hashlib
import json
import sys
import os
import time
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
os.environ['HF_HUB_ETAG_TIMEOUT']='30'
sys.path[:0]=[str(ROOT/'.semantic_deps'),str(ROOT/'.inspection_deps')]
from huggingface_hub import HfApi,hf_hub_download


def download():
    directory=ROOT/'models/bge-m3';directory.mkdir(exist_ok=True)
    path=directory/'manifest.json'
    if path.exists():
        print('Pinned BGE-M3 checkpoint already present.',flush=True)
        return
    # Resolve once, then pin: retries must not silently switch model versions.
    revision='5617a9f61b028005a4858fdac845db406aefb181'
    names=['config.json','tokenizer.json','tokenizer_config.json',
        'special_tokens_map.json','sentencepiece.bpe.model','README.md','1_Pooling/config.json','modules.json','pytorch_model.bin']
    hashes={}
    for name in names:
        print('Download BGE-M3',revision,name,flush=True)
        for attempt in range(4):
            try:
                local=Path(hf_hub_download('BAAI/bge-m3',name,revision=revision,token=False,
                    local_dir=directory,cache_dir=ROOT/'.downloads/huggingface'))
                break
            except Exception as error:
                print('Download retry',name,attempt+1,type(error).__name__,flush=True)
                if attempt==3:raise
                time.sleep(5)
        digest=hashlib.sha256()
        with local.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        hashes[name]=digest.hexdigest()
    manifest={'model':'BAAI/bge-m3','revision':revision,'license':'MIT',
        'source':'https://huggingface.co/BAAI/bge-m3','files_sha256':hashes,
        'task_data_sent':False,'external_inference_api_used':False}
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Pinned BGE-M3 downloaded',revision,flush=True)


if __name__=='__main__':download()
