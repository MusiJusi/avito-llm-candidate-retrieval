"""One-time model preparation, never invoked by the offline inference notebook.

Pinned publisher revision and official weight SHA256. A mirror transports public
files when the publisher host cannot be reached; all files are content-checked.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MODEL_ID='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
REVISION='1427fd652930e4ba29e8149678df786c240d8825'
HOST='https://hf-mirror.com'
DESTINATION=ROOT/'models/mmarco-miniLM-cross-encoder'
FILES={
 'config.json':(891,'git','43ac97f1bef372d4f1d30a32f9a84baffcf6575d'),
 'tokenizer_config.json':(435,'git','b59159d44c6bd4b1d9f7ae4f8da028da4238e06e'),
 'special_tokens_map.json':(239,'git','2ea7ad0e45a9d1d1591782ba7e29a703d0758831'),
 'tokenizer.json':(17082660,'sha256','62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626'),
 'model.safetensors':(470592698,'sha256','5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4'),
 'README.md':(2278,'git','c0f626ab888d9512bc57b224b13ff97ca93e3cb0'),
}

def digest(path,kind='sha256'):
    h=hashlib.sha1() if kind=='git' else hashlib.sha256()
    if kind=='git':
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

DESTINATION.mkdir(parents=True,exist_ok=True)
hashes={}
for name,(size,kind,expected) in FILES.items():
    target=DESTINATION/name
    if not target.exists() or target.stat().st_size!=size or digest(target,kind)!=expected:
        temporary=target.with_suffix(target.suffix+'.tmp')
        url=f'{HOST}/{MODEL_ID}/resolve/{REVISION}/{name}'
        print('Downloading',name,flush=True)
        subprocess.run(['curl.exe','--fail','--location','--retry','2','--connect-timeout','15',
                        '--max-time','450','--output',str(temporary),url],check=True)
        assert temporary.stat().st_size==size and digest(temporary,kind)==expected,name
        temporary.replace(target)
    hashes[name]=digest(target)
    print('Verified',name,size,flush=True)
manifest={'model_id':MODEL_ID,'revision':REVISION,'license':'Apache-2.0',
          'publisher_weights_sha256':FILES['model.safetensors'][2],
          'transport_host':HOST,'sha256':hashes,'external_inference_api':False}
(DESTINATION/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Local cross-encoder prepared.',flush=True)
