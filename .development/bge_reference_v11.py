"""Corpus-wide BGE score references for comparable train/inference rank features.

The reference is computed from the frozen public benchmark item corpus and
frozen BGE vectors. It contains no interactions, labels, or query IDs.
"""
from pathlib import Path
import hashlib
import json
import time

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/bge-m3-v10/vectors.json'
MANIFEST = json.loads(SOURCE.read_text(encoding='utf-8'))
FP = hashlib.sha256((MANIFEST['fingerprint'] + ':top512:fp16:v1').encode()).hexdigest()[:16]
OUT = ROOT / f'artifacts/bge-m3-v11/reference_{FP}.npy'
META = OUT.with_suffix('.json')


def build():
    if OUT.exists() and META.exists():
        print('Existing BGE reference', OUT, flush=True)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    docs = np.load(ROOT / MANIFEST['documents']['path'], mmap_mode='r')
    queries = np.load(ROOT / MANIFEST['queries']['path'], mmap_mode='r')
    assert docs.shape[1] == queries.shape[1] == 1024
    count, topk = len(queries), 512
    target = np.lib.format.open_memmap(OUT.with_suffix('.npy.partial'), mode='w+',
                                       dtype=np.float16, shape=(count, topk))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        raise RuntimeError('Corpus-wide reference requires local CUDA; no external API is used')
    start = time.perf_counter()
    with torch.inference_mode():
        d = torch.from_numpy(np.asarray(docs, dtype=np.float16)).to(device).T.contiguous()
        for first in range(0, count, 256):
            last = min(first + 256, count)
            q = torch.from_numpy(np.asarray(queries[first:last], dtype=np.float16)).to(device)
            scores = torch.mm(q, d)
            target[first:last] = scores.topk(topk, dim=1).values.cpu().numpy()
            del q, scores
            if first % 8192 == 0:
                print('BGE corpus reference', last, count,
                      'seconds', round(time.perf_counter() - start), flush=True)
    target.flush()
    del target, d
    OUT.with_suffix('.npy.partial').replace(OUT)
    texts = joblib.load(ROOT / MANIFEST['query_texts']['path'])
    assert len(texts) == count
    META.write_text(json.dumps({'fingerprint': FP, 'source': MANIFEST['fingerprint'],
                                'rows': count, 'topk': topk, 'dtype': 'float16',
                                'reference_file': OUT.relative_to(ROOT).as_posix(),
                                'seconds': round(time.perf_counter() - start, 2)}, indent=2),
                    encoding='utf-8')
    print('BGE corpus reference ready', META, flush=True)


if __name__ == '__main__':
    build()
